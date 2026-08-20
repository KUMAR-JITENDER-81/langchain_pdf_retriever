const API_BASE = window.API_BASE || "http://127.0.0.1:8000";
const API_TOKEN = window.API_TOKEN || "";

const elements = {
  health: document.querySelector("#service-health"),
  uploadForm: document.querySelector("#upload-form"),
  uploadButton: document.querySelector("#upload-button"),
  uploadStatus: document.querySelector("#upload-status"),
  uploadLimit: document.querySelector("#upload-limit"),
  uploadProgress: document.querySelector("#upload-progress"),
  fileInput: document.querySelector("#pdf-file"),
  fileName: document.querySelector("#file-name"),
  documents: document.querySelector("#documents"),
  selectedCount: document.querySelector("#selected-count"),
  chatForm: document.querySelector("#chat-form"),
  chatStatus: document.querySelector("#chat-status"),
  answer: document.querySelector("#answer"),
  answerText: document.querySelector("#answer-text"),
  answerSources: document.querySelector("#answer-sources"),
  answerWarnings: document.querySelector("#answer-warnings"),
  askButton: document.querySelector("#ask-button"),
  stopButton: document.querySelector("#stop-button"),
  question: document.querySelector("#question"),
  mode: document.querySelector("#answer-mode"),
};

const state = {
  documents: [],
  selected: new Set(),
  history: [],
  chatController: null,
  pollTimer: null,
};

function authHeaders(headers = {}) {
  const result = new Headers(headers);
  if (API_TOKEN) result.set("Authorization", `Bearer ${API_TOKEN}`);
  return result;
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: authHeaders(options.headers),
  });
  const text = await response.text();
  let body = {};
  try { body = text ? JSON.parse(text) : {}; } catch { body = { message: text }; }
  if (!response.ok) {
    throw new Error(body.error?.message || body.detail || body.message || `Request failed (${response.status})`);
  }
  return body;
}

function setStatus(element, message = "", error = false) {
  element.textContent = message;
  element.classList.toggle("error", error);
}

function formatBytes(bytes) {
  if (!Number.isFinite(Number(bytes))) return "Unknown size";
  const units = ["B", "KB", "MB", "GB"];
  let value = Number(bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

function statusLabel(document) {
  if (document.status === "processing") return document.stage?.replaceAll("_", " ") || "processing";
  return document.status || "uploaded";
}

async function loadHealth() {
  try {
    const response = await request("/health");
    const data = response.data;
    elements.health.className = "health ready";
    elements.health.innerHTML = `<span class="health-dot"></span>Service ready · ${data.embedding_provider} embeddings`;
    elements.uploadLimit.textContent = `Up to ${data.limits.upload_mb} MB · ${data.limits.pages} pages`;
  } catch (error) {
    elements.health.className = "health error";
    elements.health.innerHTML = `<span class="health-dot"></span>Service unavailable`;
    elements.uploadLimit.textContent = "Could not load upload limits";
  }
}

async function loadDocuments({ quiet = false } = {}) {
  try {
    const response = await request("/documents/");
    state.documents = response.data.documents;
    const existingIds = new Set(state.documents.map((document) => document.document_id));
    for (const selectedId of [...state.selected]) {
      if (!existingIds.has(selectedId)) state.selected.delete(selectedId);
    }
    renderDocuments();
    schedulePolling();
  } catch (error) {
    if (!quiet) setStatus(elements.uploadStatus, error.message, true);
  }
}

function renderDocuments() {
  elements.documents.replaceChildren();
  elements.documents.classList.toggle("empty", state.documents.length === 0);
  if (!state.documents.length) {
    elements.documents.textContent = "No documents uploaded yet.";
    updateSelectionCount();
    return;
  }

  for (const document of state.documents) {
    const row = window.document.createElement("div");
    row.className = "document-row";

    const checkbox = window.document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "document-select";
    checkbox.dataset.select = document.document_id;
    checkbox.disabled = document.status !== "ready";
    checkbox.checked = state.selected.has(document.document_id) && document.status === "ready";
    checkbox.setAttribute("aria-label", `Select ${document.filename || document.stored_filename}`);

    const details = window.document.createElement("div");
    const name = window.document.createElement("strong");
    name.className = "document-name";
    name.textContent = document.filename || document.stored_filename;
    const meta = window.document.createElement("div");
    meta.className = "document-meta";
    const chip = window.document.createElement("span");
    chip.className = `status-chip ${document.status}`;
    chip.textContent = statusLabel(document);
    const facts = [
      chip,
      textNode(`${document.page_count ?? "?"} pages`),
      textNode(formatBytes(document.size_bytes)),
    ];
    if (document.ocr_page_count) facts.push(textNode(`${document.ocr_page_count} OCR pages`));
    facts.forEach((fact) => meta.append(fact));
    details.append(name, meta);

    if (["queued", "processing"].includes(document.status)) {
      const progress = window.document.createElement("div");
      progress.className = "document-progress";
      const fill = window.document.createElement("span");
      fill.style.width = `${Math.max(2, Math.round(Number(document.progress || 0) * 100))}%`;
      progress.append(fill);
      details.append(progress);
    }
    if (document.error_message) {
      const error = window.document.createElement("div");
      error.className = "document-error";
      error.textContent = document.error_message;
      details.append(error);
    }

    const actions = window.document.createElement("div");
    actions.className = "document-actions";
    const openButton = actionButton("Open", "open", document.document_id, "ghost");
    const indexButton = actionButton(
      document.status === "ready" ? "Re-index" : "Index",
      "index",
      document.document_id,
      "ghost",
    );
    indexButton.disabled = ["queued", "processing"].includes(document.status);
    const deleteButton = actionButton("Delete", "delete", document.document_id, "danger");
    deleteButton.disabled = ["queued", "processing"].includes(document.status);
    actions.append(openButton, indexButton, deleteButton);

    row.append(checkbox, details, actions);
    elements.documents.append(row);
  }
  updateSelectionCount();
}

function textNode(text) {
  const span = window.document.createElement("span");
  span.textContent = text;
  return span;
}

function actionButton(label, action, id, className) {
  const button = window.document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.className = className;
  button.dataset[action] = id;
  return button;
}

function updateSelectionCount() {
  const readySelected = state.documents.filter(
    (document) => document.status === "ready" && state.selected.has(document.document_id),
  ).length;
  elements.selectedCount.textContent = `${readySelected} selected`;
}

function schedulePolling() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  const hasActiveJobs = state.documents.some((document) => ["queued", "processing"].includes(document.status));
  if (hasActiveJobs) {
    state.pollTimer = setTimeout(() => loadDocuments({ quiet: true }), 1400);
  }
}

function uploadPdf(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/upload/`);
    if (API_TOKEN) xhr.setRequestHeader("Authorization", `Bearer ${API_TOKEN}`);
    xhr.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      const percentage = Math.round((event.loaded / event.total) * 100);
      elements.uploadProgress.querySelector("span").style.width = `${percentage}%`;
      setStatus(elements.uploadStatus, `Uploading… ${percentage}%`);
    });
    xhr.addEventListener("load", () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText || "{}"); } catch { body = {}; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.error?.message || body.detail || body.message || "Upload failed"));
    });
    xhr.addEventListener("error", () => reject(new Error("Could not reach the upload service")));
    const formData = new FormData();
    formData.append("file", file);
    xhr.send(formData);
  });
}

async function queueIndex(documentId, force = false) {
  await request(`/documents/${documentId}/index?force=${force}`, { method: "POST" });
  setStatus(elements.uploadStatus, "Indexing started. OCR may take longer for scanned pages.");
  await loadDocuments({ quiet: true });
}

elements.fileInput.addEventListener("change", () => {
  elements.fileName.textContent = elements.fileInput.files[0]?.name || "Searchable, scanned, or handwritten";
});

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = elements.fileInput.files[0];
  if (!file) return;
  elements.uploadButton.disabled = true;
  elements.uploadProgress.classList.remove("hidden");
  elements.uploadProgress.querySelector("span").style.width = "0";
  try {
    const response = await uploadPdf(file);
    const document = response.data;
    state.selected.add(document.document_id);
    setStatus(elements.uploadStatus, response.message);
    await loadDocuments({ quiet: true });
    if (document.status !== "ready") await queueIndex(document.document_id, false);
    elements.uploadForm.reset();
    elements.fileName.textContent = "Searchable, scanned, or handwritten";
  } catch (error) {
    setStatus(elements.uploadStatus, error.message, true);
  } finally {
    elements.uploadButton.disabled = false;
    setTimeout(() => elements.uploadProgress.classList.add("hidden"), 500);
  }
});

elements.documents.addEventListener("change", (event) => {
  const id = event.target.dataset.select;
  if (!id) return;
  if (event.target.checked) state.selected.add(id);
  else state.selected.delete(id);
  updateSelectionCount();
});

elements.documents.addEventListener("click", async (event) => {
  const indexId = event.target.dataset.index;
  const deleteId = event.target.dataset.delete;
  const openId = event.target.dataset.open;
  try {
    if (indexId) await queueIndex(indexId, true);
    if (deleteId) {
      const item = state.documents.find((document) => document.document_id === deleteId);
      if (!window.confirm(`Delete ${item?.filename || "this document"}?`)) return;
      await request(`/documents/${deleteId}`, { method: "DELETE" });
      state.selected.delete(deleteId);
      setStatus(elements.uploadStatus, "Document deleted.");
      await loadDocuments({ quiet: true });
    }
    if (openId) await openPdf(openId, 1);
  } catch (error) {
    setStatus(elements.uploadStatus, error.message, true);
  }
});

document.querySelector("#refresh-documents").addEventListener("click", () => loadDocuments());
document.querySelector("#select-ready").addEventListener("click", () => {
  const ready = state.documents.filter((document) => document.status === "ready");
  const allSelected = ready.length && ready.every((document) => state.selected.has(document.document_id));
  ready.forEach((document) => {
    if (allSelected) state.selected.delete(document.document_id);
    else state.selected.add(document.document_id);
  });
  renderDocuments();
});

elements.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const selectedIds = state.documents
    .filter((document) => document.status === "ready" && state.selected.has(document.document_id))
    .map((document) => document.document_id);
  if (!selectedIds.length) {
    setStatus(elements.chatStatus, "Select at least one ready document.", true);
    return;
  }

  const question = elements.question.value.trim();
  const mode = elements.mode.value;
  const k = mode === "quick" ? 4 : mode === "deep" ? 8 : 5;
  state.chatController = new AbortController();
  elements.askButton.disabled = true;
  elements.stopButton.classList.remove("hidden");
  elements.answer.classList.remove("hidden");
  elements.answerText.textContent = "";
  elements.answerSources.replaceChildren();
  elements.answerWarnings.classList.add("hidden");
  setStatus(elements.chatStatus, "Searching selected documents…");

  try {
    const completedAnswer = await streamChat({
      question,
      mode,
      k,
      document_ids: selectedIds,
      history: state.history.slice(-12),
    }, state.chatController.signal);
    state.history.push({ role: "user", content: question });
    state.history.push({ role: "assistant", content: completedAnswer });
    state.history = state.history.slice(-12);
    setStatus(elements.chatStatus, "Answer complete.");
  } catch (error) {
    setStatus(
      elements.chatStatus,
      error.name === "AbortError" ? "Answer stopped." : error.message,
      error.name !== "AbortError",
    );
  } finally {
    state.chatController = null;
    elements.askButton.disabled = false;
    elements.stopButton.classList.add("hidden");
  }
});

elements.stopButton.addEventListener("click", () => state.chatController?.abort());

async function streamChat(payload, signal) {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error?.message || body.detail || body.message || "Chat request failed");
  }
  if (!response.body) throw new Error("Streaming is not supported by this browser");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replaceAll("\r\n", "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseSseBlock(block);
      if (!parsed) continue;
      if (parsed.event === "sources") {
        renderSources(parsed.data.sources || []);
        renderWarnings(parsed.data.warnings || []);
        setStatus(elements.chatStatus, "Generating answer…");
      } else if (parsed.event === "token") {
        answer += parsed.data.text || "";
        elements.answerText.textContent = answer;
      } else if (parsed.event === "error") {
        throw new Error(parsed.data.message || "Answer generation failed");
      } else if (parsed.event === "done") {
        answer = parsed.data.answer || answer;
      }
    }
    if (done) break;
  }
  return answer;
}

function parseSseBlock(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try { return { event, data: JSON.parse(dataLines.join("\n")) }; }
  catch { return null; }
}

function renderWarnings(warnings) {
  elements.answerWarnings.classList.toggle("hidden", warnings.length === 0);
  elements.answerWarnings.textContent = warnings.join(" · ");
}

function renderSources(sources) {
  elements.answerSources.replaceChildren();
  for (const source of sources) {
    const card = window.document.createElement("div");
    card.className = "source-card";
    const heading = window.document.createElement("div");
    heading.className = "source-heading";
    const title = window.document.createElement("span");
    title.className = "source-title";
    title.textContent = `Source ${source.source_id} · ${source.filename || "PDF"} · Page ${source.page}`;
    const button = window.document.createElement("button");
    button.type = "button";
    button.textContent = "Open page";
    button.addEventListener("click", () => openPdf(source.document_id, source.page));
    heading.append(title, button);
    const snippet = window.document.createElement("p");
    snippet.textContent = source.snippet;
    const badges = window.document.createElement("div");
    badges.className = "source-badges";
    const relevance = Math.max(0, Math.min(Number(source.relevance || 0), 1));
    badges.append(textBadge(`${Math.round(relevance * 100)}% relevance`));
    if (source.extraction_method && source.extraction_method !== "native") {
      badges.append(textBadge(source.extraction_method.toUpperCase()));
    }
    if (source.handwritten) badges.append(textBadge("Handwritten"));
    card.append(heading, snippet, badges);
    elements.answerSources.append(card);
  }
}

function textBadge(text) {
  const badge = window.document.createElement("span");
  badge.className = "badge";
  badge.textContent = text;
  return badge;
}

async function openPdf(documentId, page) {
  const popup = window.open("about:blank", "_blank");
  try {
    const response = await fetch(`${API_BASE}/documents/${documentId}/file`, {
      headers: authHeaders(),
    });
    if (!response.ok) throw new Error("Could not open PDF");
    const url = URL.createObjectURL(await response.blob());
    if (popup) popup.location.href = `${url}#page=${page || 1}`;
    else window.location.href = `${url}#page=${page || 1}`;
    setTimeout(() => URL.revokeObjectURL(url), 120000);
  } catch (error) {
    if (popup) popup.close();
    setStatus(elements.uploadStatus, error.message, true);
  }
}

Promise.all([loadHealth(), loadDocuments()]);
