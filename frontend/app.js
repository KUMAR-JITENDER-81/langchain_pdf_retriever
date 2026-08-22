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
  filePicker: document.querySelector(".file-picker"),
  fileName: document.querySelector("#file-name"),
  documents: document.querySelector("#documents"),
  selectedCount: document.querySelector("#selected-count"),
  chatForm: document.querySelector("#chat-form"),
  chatStatus: document.querySelector("#chat-status"),
  answer: document.querySelector("#answer"),
  answerText: document.querySelector("#answer-text"),
  answerMeta: document.querySelector("#answer-meta"),
  answerDiagnostics: document.querySelector("#answer-diagnostics"),
  answerSources: document.querySelector("#answer-sources"),
  answerWarnings: document.querySelector("#answer-warnings"),
  askButton: document.querySelector("#ask-button"),
  stopButton: document.querySelector("#stop-button"),
  question: document.querySelector("#question"),
  mode: document.querySelector("#answer-mode"),
  task: document.querySelector("#answer-task"),
  languageField: document.querySelector("#language-field"),
  responseLanguage: document.querySelector("#response-language"),
  copyAnswer: document.querySelector("#copy-answer"),
  clearAnswer: document.querySelector("#clear-answer"),
  suggestions: document.querySelector(".suggestions"),
  localAiPanel: document.querySelector("#local-ai-panel"),
  localAiTitle: document.querySelector("#local-ai-title"),
  localAiDetail: document.querySelector("#local-ai-detail"),
  warmupAi: document.querySelector("#warmup-ai"),
  answerFeedback: document.querySelector("#answer-feedback"),
  feedbackDetails: document.querySelector("#feedback-details"),
  feedbackReasons: document.querySelector("#feedback-reasons"),
  feedbackComment: document.querySelector("#feedback-comment"),
  feedbackStatus: document.querySelector("#feedback-status"),
  submitFeedback: document.querySelector("#submit-feedback"),
  previewOverlay: document.querySelector("#preview-overlay"),
  previewTitle: document.querySelector("#preview-title"),
  previewDetail: document.querySelector("#preview-detail"),
  previewStatus: document.querySelector("#preview-status"),
  previewImage: document.querySelector("#preview-image"),
  previewClose: document.querySelector("#preview-close"),
  previewOpenFull: document.querySelector("#preview-open-full"),
  qualityMetrics: document.querySelector("#quality-metrics"),
  qualityDetail: document.querySelector("#quality-detail"),
  refreshQuality: document.querySelector("#refresh-quality"),
};

const state = {
  documents: [],
  selected: new Set(),
  history: [],
  chatController: null,
  pollTimer: null,
  healthData: null,
  answerWarnings: [],
  selectionInitialized: false,
  lastAnswer: "",
  lastSources: [],
  currentAnswerId: "",
  feedbackRating: "",
  feedbackReasons: new Set(),
  currentPreviewSource: null,
  previewObjectUrl: "",
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
  if (document.status === "processing") {
    const labels = {
      starting: "starting",
      extracting_native_text: "reading text",
      extracting_text: "reading pages",
      ocr: "running OCR",
      chunking: "splitting text",
      embedding: "building search index",
      indexing: "saving index",
      cancelling: "cancelling",
    };
    return labels[document.stage] || document.stage?.replaceAll("_", " ") || "processing";
  }
  return document.status || "uploaded";
}

async function loadHealth() {
  try {
    const response = await request("/health");
    const data = response.data;
    state.healthData = data;
    const balancedMode = data.answer_modes?.balanced;
    const deepMode = data.answer_modes?.deep;
    const generationLabel = data.ollama?.available
      ? "Ollama connected"
      : "evidence fallback ready";
    elements.health.className = "health ready";
    const dot = window.document.createElement("span");
    dot.className = "health-dot";
    elements.health.replaceChildren(
      dot,
      window.document.createTextNode(`Free local mode · ${generationLabel}`),
    );
    elements.uploadLimit.textContent = `Up to ${data.limits.upload_mb} MB · ${data.limits.pages} pages`;
    renderOllamaStatus(data);
    const quickOption = elements.mode.querySelector('option[value="quick"]');
    const balancedOption = elements.mode.querySelector('option[value="balanced"]');
    const deepOption = elements.mode.querySelector('option[value="deep"]');
    if (quickOption) {
      quickOption.textContent = "Quick — instant cited evidence (no model wait)";
    }
    if (balancedOption) {
      balancedOption.disabled = balancedMode?.ready === false;
      balancedOption.textContent = balancedMode?.engine === "ollama"
        ? `Balanced — ${balancedMode.model} local AI${balancedMode.using_fallback_model ? " (fallback)" : ""}`
        : "Balanced — fast structured evidence";
    }
    if (deepOption) {
      deepOption.disabled = deepMode?.ready === false;
      deepOption.textContent = deepMode?.engine === "ollama"
        ? `Deep — ${deepMode.model} synthesis${deepMode.using_fallback_model ? " (fallback)" : " (slower)"}`
        : "Deep — evidence fallback";
    }
  } catch (error) {
    elements.health.className = "health error";
    elements.health.innerHTML = `<span class="health-dot"></span>Service unavailable`;
    elements.uploadLimit.textContent = "Could not load upload limits";
    elements.localAiPanel.className = "local-ai-panel error";
    elements.localAiTitle.textContent = "Backend unavailable";
    elements.localAiDetail.textContent = "Start the API, then this panel will reconnect automatically.";
    elements.warmupAi.disabled = true;
  }
}

function renderOllamaStatus(data) {
  const ollama = data.ollama || {};
  const warmup = data.ollama_warmup || {};
  if (!ollama.available) {
    elements.localAiPanel.className = "local-ai-panel error";
    elements.localAiTitle.textContent = "Ollama is offline";
    elements.localAiDetail.textContent = "Quick answers still work through the cited evidence fallback.";
    elements.warmupAi.textContent = "Ollama offline";
    elements.warmupAi.disabled = true;
    return;
  }

  const installedCount = Array.isArray(ollama.installed_models) ? ollama.installed_models.length : 0;
  const loadedModels = Array.isArray(ollama.loaded_models) ? ollama.loaded_models : [];
  const balancedModel = data.answer_modes?.balanced?.model || ollama.effective_fast_model;
  const deepModel = data.answer_modes?.deep?.model || ollama.effective_deep_model;
  const balancedLoaded = modelListIncludes(loadedModels, balancedModel);
  const allModelsReady = Boolean(ollama.all_required_models_installed);
  elements.localAiPanel.className = `local-ai-panel ${allModelsReady ? "ready" : "warning"}`;
  elements.localAiTitle.textContent = `Ollama connected · ${installedCount} local model${installedCount === 1 ? "" : "s"} installed`;

  const details = [
    `Balanced: ${balancedModel}`,
    `Deep: ${deepModel}`,
    ollama.ocr_model_installed
      ? `Handwriting OCR: ${ollama.ocr_model}`
      : `Handwriting OCR model missing: ${ollama.ocr_model}`,
  ];
  const queue = data.index_queue || {};
  if (queue.active) details.push(`Index jobs: ${queue.active} active`);
  if (data.reranker?.installed) details.push("Semantic reranker ready");
  elements.localAiDetail.textContent = details.join(" · ");

  const warming = warmup.state === "warming";
  elements.warmupAi.disabled = warming || balancedLoaded;
  elements.warmupAi.textContent = warming
    ? "Warming up…"
    : balancedLoaded
      ? "AI ready"
      : "Warm up AI";
}

function modelListIncludes(models, target) {
  const normalize = (value) => String(value || "").replace(/:latest$/, "");
  return models.some((model) => normalize(model) === normalize(target));
}

async function loadDocuments({ quiet = false } = {}) {
  try {
    const response = await request("/documents/");
    state.documents = response.data.documents;
    const existingIds = new Set(state.documents.map((document) => document.document_id));
    for (const selectedId of [...state.selected]) {
      if (!existingIds.has(selectedId)) state.selected.delete(selectedId);
    }
    if (!state.selectionInitialized) {
      const firstReady = state.documents.find((document) => document.status === "ready");
      if (!state.selected.size && firstReady) state.selected.add(firstReady.document_id);
      state.selectionInitialized = true;
    }
    renderDocuments();
    schedulePolling();
  } catch (error) {
    if (!quiet) setStatus(elements.uploadStatus, error.message, true);
  }
}

async function loadQualitySummary() {
  try {
    const response = await request("/quality/summary");
    const data = response.data || {};
    const values = [
      [String(data.answer_count || 0), "Answers measured"],
      [
        data.helpful_rate == null ? "No ratings" : `${Math.round(Number(data.helpful_rate) * 100)}%`,
        "Helpful feedback",
      ],
      [formatDuration(data.average_total_ms || 0), "Average response"],
      [`${Math.round(Number(data.average_quality_score || 0) * 100)}%`, "Grounding checks"],
    ];
    elements.qualityMetrics.replaceChildren();
    for (const [value, label] of values) {
      const card = window.document.createElement("div");
      const strong = window.document.createElement("strong");
      strong.textContent = value;
      const span = window.document.createElement("span");
      span.textContent = label;
      card.append(strong, span);
      elements.qualityMetrics.append(card);
    }
    const reasons = Object.entries(data.failure_reasons || {});
    elements.qualityDetail.textContent = reasons.length
      ? `Most reported issue: ${reasonLabel(reasons[0][0])} (${reasons[0][1]}). Metrics stay in the local SQLite database.`
      : "Automatic grounding checks measure citations and retrieval confidence; they are not a guarantee of factual correctness.";
  } catch (error) {
    elements.qualityDetail.textContent = `Quality metrics unavailable: ${error.message}`;
  }
}

function reasonLabel(value) {
  return String(value || "").replaceAll("_", " ");
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
    if (document.handwritten_page_count) facts.push(textNode(`${document.handwritten_page_count} handwritten`));
    if (document.table_count) facts.push(textNode(`${document.table_count} tables`));
    if (document.low_quality_page_count) facts.push(textNode(`${document.low_quality_page_count} pages need review`));
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
      const legacyProviderFailure = [
        "provider_quota_exceeded",
        "openai_rate_limited",
        "openai_connection_error",
      ].includes(document.error_code);
      error.textContent = legacyProviderFailure && state.healthData?.cost_mode === "free-local"
        ? "Previous paid-provider indexing failed. Index again to migrate this PDF locally."
        : document.error_message;
      details.append(error);
    }

    const actions = window.document.createElement("div");
    actions.className = "document-actions";
    const openButton = actionButton("Open", "open", document.document_id, "ghost");
    const indexButton = actionButton(
      document.status === "ready" ? "Refresh index" : "Index locally",
      "index",
      document.document_id,
      "ghost",
    );
    indexButton.disabled = ["queued", "processing"].includes(document.status);
    const deleteButton = actionButton("Delete", "delete", document.document_id, "danger");
    deleteButton.disabled = ["queued", "processing"].includes(document.status);
    actions.append(openButton, indexButton);
    if (["queued", "processing"].includes(document.status)) {
      const cancelButton = actionButton("Cancel", "cancelIndex", document.document_id, "danger");
      actions.append(cancelButton);
    }
    actions.append(deleteButton);

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
  if (!state.chatController) elements.askButton.disabled = readySelected === 0;
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

function validatePdf(file) {
  if (!file) return "Choose a PDF first.";
  if (!file.name.toLowerCase().endsWith(".pdf")) return "Only PDF files are supported.";
  const maxMegabytes = Number(state.healthData?.limits?.upload_mb || 100);
  if (file.size > maxMegabytes * 1024 * 1024) {
    return `This file is ${formatBytes(file.size)}. The current limit is ${maxMegabytes} MB.`;
  }
  return "";
}

elements.fileInput.addEventListener("change", () => {
  const file = elements.fileInput.files[0];
  const error = validatePdf(file);
  if (file && error) {
    elements.fileInput.value = "";
    elements.fileName.textContent = "Searchable, scanned, or handwritten";
    setStatus(elements.uploadStatus, error, true);
    return;
  }
  elements.fileName.textContent = file?.name || "Searchable, scanned, or handwritten";
  setStatus(elements.uploadStatus, file ? `${formatBytes(file.size)} ready to upload.` : "");
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.filePicker.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.filePicker.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.filePicker.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.filePicker.classList.remove("dragging");
  });
}
elements.filePicker.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  const error = validatePdf(file);
  if (error) {
    setStatus(elements.uploadStatus, error, true);
    return;
  }
  const transfer = new DataTransfer();
  transfer.items.add(file);
  elements.fileInput.files = transfer.files;
  elements.fileInput.dispatchEvent(new Event("change"));
});

elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = elements.fileInput.files[0];
  const validationError = validatePdf(file);
  if (validationError) {
    setStatus(elements.uploadStatus, validationError, true);
    return;
  }
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
  const cancelIndexId = event.target.dataset.cancelIndex;
  try {
    if (indexId) {
      await queueIndex(indexId, false);
    }
    if (deleteId) {
      const item = state.documents.find((document) => document.document_id === deleteId);
      if (!window.confirm(`Delete ${item?.filename || "this document"}?`)) return;
      await request(`/documents/${deleteId}`, { method: "DELETE" });
      state.selected.delete(deleteId);
      setStatus(elements.uploadStatus, "Document deleted.");
      await loadDocuments({ quiet: true });
    }
    if (openId) await openPdf(openId, 1);
    if (cancelIndexId) {
      await request(`/documents/${cancelIndexId}/index/cancel`, { method: "POST" });
      setStatus(elements.uploadStatus, "Cancelling indexing. The current OCR page may finish first.");
      await loadDocuments({ quiet: true });
    }
  } catch (error) {
    setStatus(elements.uploadStatus, error.message, true);
  }
});

document.querySelector("#refresh-documents").addEventListener("click", () => loadDocuments());
document.querySelector("#reindex-ready").addEventListener("click", async (event) => {
  const readyIds = state.documents
    .filter((document) => document.status === "ready")
    .map((document) => document.document_id);
  if (!readyIds.length) {
    setStatus(elements.uploadStatus, "There are no ready documents to refresh.");
    return;
  }
  event.currentTarget.disabled = true;
  try {
    const response = await request("/documents/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_ids: readyIds, force: true }),
    });
    setStatus(elements.uploadStatus, response.message);
    await loadDocuments({ quiet: true });
  } catch (error) {
    setStatus(elements.uploadStatus, error.message, true);
  } finally {
    event.currentTarget.disabled = false;
  }
});
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
  const task = elements.task.value;
  const k = mode === "quick" ? 4 : mode === "deep" ? 12 : 8;
  const startedAt = performance.now();
  state.chatController = new AbortController();
  elements.askButton.disabled = true;
  elements.stopButton.classList.remove("hidden");
  elements.answer.classList.remove("hidden");
  elements.answerText.textContent = "";
  elements.answerMeta.textContent = "";
  elements.answerDiagnostics.textContent = "";
  elements.answerSources.replaceChildren();
  elements.answerWarnings.classList.add("hidden");
  state.answerWarnings = [];
  state.lastAnswer = "";
  state.lastSources = [];
  resetFeedback();
  setStatus(elements.chatStatus, "Searching selected documents…");

  try {
    const completedAnswer = await streamChat({
      question,
      mode,
      task,
      response_language: elements.responseLanguage.value.trim() || "English",
      k,
      document_ids: selectedIds,
      history: state.history.slice(-12),
    }, state.chatController.signal);
    state.history.push({ role: "user", content: question });
    state.history.push({ role: "assistant", content: completedAnswer });
    state.history = state.history.slice(-12);
    state.lastAnswer = completedAnswer;
    const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
    setStatus(elements.chatStatus, `Answer complete in ${elapsedSeconds} seconds.`);
  } catch (error) {
    setStatus(
      elements.chatStatus,
      error.name === "AbortError" ? "Answer stopped." : error.message,
      error.name !== "AbortError",
    );
  } finally {
    state.chatController = null;
    elements.stopButton.classList.add("hidden");
    updateSelectionCount();
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
  let receivedFirstToken = false;
  let activeModel = "";
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
        state.currentAnswerId = parsed.data.answer_id || "";
        renderSources(parsed.data.sources || []);
        renderWarnings(parsed.data.warnings || []);
        activeModel = parsed.data.model || "";
        renderAnswerMeta(parsed.data);
        setStatus(
          elements.chatStatus,
          parsed.data.engine === "ollama"
            ? `Generating privately with ${activeModel || "Ollama"}…`
            : "Building an instant answer from cited evidence…",
        );
      } else if (parsed.event === "warning") {
        renderWarnings([...state.answerWarnings, parsed.data.message].filter(Boolean));
      } else if (parsed.event === "reset") {
        answer = "";
        elements.answerText.replaceChildren();
      } else if (parsed.event === "token") {
        if (!receivedFirstToken) {
          receivedFirstToken = true;
          setStatus(
            elements.chatStatus,
            activeModel ? `Receiving answer from ${activeModel}…` : "Receiving answer…",
          );
        }
        answer += parsed.data.text || "";
        elements.answerText.textContent = answer;
      } else if (parsed.event === "error") {
        throw new Error(parsed.data.message || "Answer generation failed");
      } else if (parsed.event === "done") {
        answer = parsed.data.answer || answer;
        state.currentAnswerId = parsed.data.answer_id || state.currentAnswerId;
        renderDiagnostics(parsed.data.diagnostics || {});
        renderAnswerText(answer);
        elements.answerFeedback.classList.toggle("hidden", !state.currentAnswerId);
      }
    }
    if (done) break;
  }
  return answer;
}

function renderAnswerMeta(data) {
  const parts = [];
  if (data.engine === "ollama") parts.push(`Local AI: ${data.model || "Ollama"}`);
  else if (data.engine === "extractive") parts.push("Instant evidence mode");
  const profile = data.document_profile || {};
  if (data.task && data.task !== "answer") parts.push(`Task: ${reasonLabel(data.task)}`);
  if (profile.type) parts.push(`Detected: ${profile.type}`);
  if (Array.isArray(profile.sections) && profile.sections.length) {
    parts.push(`Sections: ${profile.sections.slice(0, 4).join(", ")}`);
  }
  elements.answerMeta.textContent = parts.join(" · ");
}

function renderDiagnostics(diagnostics) {
  const parts = [];
  if (diagnostics.cache_hit) parts.push("Instant local cache");
  if (Number.isFinite(Number(diagnostics.retrieval_ms))) {
    parts.push(`Search ${formatDuration(diagnostics.retrieval_ms)}`);
  }
  if (Number.isFinite(Number(diagnostics.generation_ms))) {
    parts.push(`Answer ${formatDuration(diagnostics.generation_ms)}`);
  }
  if (Number.isFinite(Number(diagnostics.citation_coverage))) {
    parts.push(`${Math.round(Number(diagnostics.citation_coverage) * 100)}% claims cited`);
  }
  if (diagnostics.ranker && !diagnostics.cache_hit) parts.push(`Ranker: ${diagnostics.ranker}`);
  elements.answerDiagnostics.textContent = parts.join(" · ");
}

function formatDuration(milliseconds) {
  const value = Number(milliseconds);
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(1)} s`;
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

function renderAnswerText(text) {
  elements.answerText.replaceChildren();
  let activeList = null;
  for (const rawLine of String(text || "").split("\n")) {
    const line = rawLine.trim();
    if (!line) {
      activeList = null;
      continue;
    }
    if (line.startsWith("## ")) {
      const heading = window.document.createElement("h3");
      appendAnswerContent(heading, cleanInlineMarkdown(line.slice(3)));
      elements.answerText.append(heading);
      activeList = null;
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      if (!activeList) {
        activeList = window.document.createElement("ul");
        elements.answerText.append(activeList);
      }
      const item = window.document.createElement("li");
      appendAnswerContent(item, cleanInlineMarkdown(line.replace(/^[-*]\s+/, "")));
      activeList.append(item);
      continue;
    }
    const paragraph = window.document.createElement("p");
    appendAnswerContent(paragraph, cleanInlineMarkdown(line.replace(/^#{1,3}\s+/, "")));
    elements.answerText.append(paragraph);
    activeList = null;
  }
}

function appendAnswerContent(parent, text) {
  const citationPattern = /\[Source\s+(\d+)\]/gi;
  let cursor = 0;
  for (const match of text.matchAll(citationPattern)) {
    parent.append(window.document.createTextNode(text.slice(cursor, match.index)));
    const sourceId = Number(match[1]);
    const source = state.lastSources.find((item) => Number(item.source_id) === sourceId);
    const citation = window.document.createElement("button");
    citation.type = "button";
    citation.className = "citation-link";
    citation.textContent = `[Source ${sourceId}]`;
    citation.title = source
      ? `Open ${source.filename || "PDF"}, page ${source.page || 1}`
      : `Source ${sourceId}`;
    citation.disabled = !source;
    if (source) {
      citation.addEventListener("click", () => showSourcePreview(source));
    }
    parent.append(citation);
    cursor = Number(match.index) + match[0].length;
  }
  parent.append(window.document.createTextNode(text.slice(cursor)));
}

function cleanInlineMarkdown(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/`(.+?)`/g, "$1");
}

function renderWarnings(warnings) {
  state.answerWarnings = [...new Set(warnings)];
  elements.answerWarnings.classList.toggle("hidden", state.answerWarnings.length === 0);
  elements.answerWarnings.textContent = state.answerWarnings.join(" · ");
}

function renderSources(sources) {
  state.lastSources = sources;
  elements.answerSources.replaceChildren();
  for (const source of sources) {
    const card = window.document.createElement("div");
    card.className = "source-card";
    const heading = window.document.createElement("div");
    heading.className = "source-heading";
    const title = window.document.createElement("span");
    title.className = "source-title";
    title.textContent = `Source ${source.source_id} · ${source.filename || "PDF"} · Page ${source.page}`;
    const actions = window.document.createElement("div");
    actions.className = "source-actions";
    const previewButton = window.document.createElement("button");
    previewButton.type = "button";
    previewButton.textContent = "Preview";
    previewButton.addEventListener("click", () => showSourcePreview(source));
    const openButton = window.document.createElement("button");
    openButton.type = "button";
    openButton.className = "ghost";
    openButton.textContent = "Full PDF";
    openButton.addEventListener("click", () => openPdf(source.document_id, source.page));
    actions.append(previewButton, openButton);
    heading.append(title, actions);
    const snippet = window.document.createElement("p");
    snippet.textContent = source.snippet;
    const badges = window.document.createElement("div");
    badges.className = "source-badges";
    const relevance = Math.max(0, Math.min(Number(source.relevance || 0), 1));
    badges.append(textBadge(`${Math.round(relevance * 100)}% relevance`));
    if (source.extraction_method && source.extraction_method !== "native") {
      badges.append(textBadge(source.extraction_method.toUpperCase()));
    }
    if (source.ocr_confidence != null && Number.isFinite(Number(source.ocr_confidence))) {
      badges.append(textBadge(`${Math.round(Number(source.ocr_confidence) * 100)}% OCR confidence`));
    }
    if (source.text_quality != null && Number.isFinite(Number(source.text_quality)) && Number(source.text_quality) < 0.5) {
      badges.append(textBadge("Check scan quality"));
    }
    if (source.handwritten) badges.append(textBadge("Handwritten"));
    if (source.content_type === "table") badges.append(textBadge("Table evidence"));
    if (Array.isArray(source.bbox)) badges.append(textBadge("Exact region available"));
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

elements.copyAnswer.addEventListener("click", async () => {
  if (!state.lastAnswer) return;
  try {
    await navigator.clipboard.writeText(state.lastAnswer);
    setStatus(elements.chatStatus, "Answer copied.");
  } catch {
    setStatus(elements.chatStatus, "Could not copy the answer automatically.", true);
  }
});

elements.clearAnswer.addEventListener("click", () => {
  state.lastAnswer = "";
  state.answerWarnings = [];
  elements.answer.classList.add("hidden");
  elements.answerText.replaceChildren();
  elements.answerMeta.textContent = "";
  elements.answerDiagnostics.textContent = "";
  elements.answerSources.replaceChildren();
  elements.answerWarnings.classList.add("hidden");
  resetFeedback();
  setStatus(elements.chatStatus, "");
  elements.question.focus();
});

function resetFeedback() {
  state.currentAnswerId = "";
  state.feedbackRating = "";
  state.feedbackReasons = new Set();
  elements.answerFeedback.classList.add("hidden");
  elements.feedbackDetails.classList.add("hidden");
  elements.feedbackComment.value = "";
  setStatus(elements.feedbackStatus, "");
  for (const button of elements.answerFeedback.querySelectorAll("button")) {
    button.classList.remove("selected");
    button.disabled = false;
    if (button.dataset.feedbackReason) button.setAttribute("aria-pressed", "false");
  }
}

elements.answerFeedback.addEventListener("click", async (event) => {
  const rating = event.target.dataset.feedbackRating;
  const reason = event.target.dataset.feedbackReason;
  if (rating) {
    state.feedbackRating = rating;
    for (const button of elements.answerFeedback.querySelectorAll("[data-feedback-rating]")) {
      const selected = button.dataset.feedbackRating === rating;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    }
    elements.feedbackDetails.classList.toggle("hidden", rating !== "not_helpful");
    if (rating === "helpful") await saveAnswerFeedback();
  }
  if (reason) {
    if (state.feedbackReasons.has(reason)) state.feedbackReasons.delete(reason);
    else state.feedbackReasons.add(reason);
    const selected = state.feedbackReasons.has(reason);
    event.target.classList.toggle("selected", selected);
    event.target.setAttribute("aria-pressed", String(selected));
  }
});

elements.submitFeedback.addEventListener("click", () => saveAnswerFeedback());

async function saveAnswerFeedback() {
  if (!state.currentAnswerId || !state.feedbackRating) return;
  elements.submitFeedback.disabled = true;
  try {
    await request("/quality/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        answer_id: state.currentAnswerId,
        rating: state.feedbackRating,
        reasons: [...state.feedbackReasons],
        comment: elements.feedbackComment.value.trim(),
      }),
    });
    setStatus(elements.feedbackStatus, "Feedback saved locally. Thank you.");
    loadQualitySummary();
    if (state.feedbackRating === "helpful") {
      elements.feedbackDetails.classList.add("hidden");
    }
  } catch (error) {
    setStatus(elements.feedbackStatus, error.message, true);
  } finally {
    elements.submitFeedback.disabled = false;
  }
}

async function showSourcePreview(source) {
  state.currentPreviewSource = source;
  closePreviewImage();
  elements.previewOverlay.classList.remove("hidden");
  document.body.classList.add("preview-open");
  elements.previewTitle.textContent = `Source ${source.source_id} · ${source.filename || "PDF"}`;
  elements.previewDetail.textContent = `Page ${source.page || 1}${source.content_type === "table" ? " · table" : ""}`;
  elements.previewStatus.textContent = Array.isArray(source.bbox)
    ? "Loading the highlighted evidence region…"
    : "Loading page preview; exact coordinates are unavailable for this OCR source.";
  elements.previewImage.classList.add("hidden");
  const parameters = new URLSearchParams();
  if (Array.isArray(source.bbox) && source.bbox.length === 4) {
    ["x0", "y0", "x1", "y1"].forEach((name, index) => parameters.set(name, source.bbox[index]));
  }
  const suffix = parameters.size ? `?${parameters}` : "";
  try {
    const response = await fetch(
      `${API_BASE}/documents/${source.document_id}/pages/${source.page || 1}/preview${suffix}`,
      { headers: authHeaders() },
    );
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error?.message || "Could not render this page");
    }
    state.previewObjectUrl = URL.createObjectURL(await response.blob());
    elements.previewImage.src = state.previewObjectUrl;
    elements.previewImage.classList.remove("hidden");
    elements.previewStatus.textContent = response.headers.get("X-Evidence-Highlighted") === "true"
      ? "The green box marks the evidence used in the answer."
      : "Showing the cited page. Open the full PDF to inspect surrounding content.";
  } catch (error) {
    elements.previewStatus.textContent = error.message;
  }
}

function closePreviewImage() {
  if (state.previewObjectUrl) URL.revokeObjectURL(state.previewObjectUrl);
  state.previewObjectUrl = "";
  elements.previewImage.removeAttribute("src");
}

function closeSourcePreview() {
  closePreviewImage();
  state.currentPreviewSource = null;
  elements.previewOverlay.classList.add("hidden");
  document.body.classList.remove("preview-open");
}

elements.previewClose.addEventListener("click", closeSourcePreview);
elements.previewOverlay.addEventListener("click", (event) => {
  if (event.target === elements.previewOverlay) closeSourcePreview();
});
elements.previewOpenFull.addEventListener("click", () => {
  const source = state.currentPreviewSource;
  if (source) openPdf(source.document_id, source.page);
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !elements.previewOverlay.classList.contains("hidden")) {
    closeSourcePreview();
  }
});

elements.refreshQuality.addEventListener("click", async () => {
  elements.refreshQuality.disabled = true;
  await loadQualitySummary();
  elements.refreshQuality.disabled = false;
});

elements.suggestions.addEventListener("click", (event) => {
  const question = event.target.dataset.question;
  if (!question) return;
  elements.question.value = question;
  if (event.target.dataset.task) {
    elements.task.value = event.target.dataset.task;
    elements.task.dispatchEvent(new Event("change"));
  }
  elements.question.focus();
});

elements.task.addEventListener("change", () => {
  const task = elements.task.value;
  elements.languageField.classList.toggle("hidden", task !== "translate");
  const placeholders = {
    answer: "What does the document say about…?",
    summary: "What should the summary focus on?",
    compare: "Which topics should be compared?",
    extract: "Which fields or table should be extracted?",
    quiz: "Which topics should the quiz cover?",
    translate: "Which information should be translated?",
  };
  elements.question.placeholder = placeholders[task] || placeholders.answer;
});

elements.warmupAi.addEventListener("click", async () => {
  elements.warmupAi.disabled = true;
  elements.warmupAi.textContent = "Warming up…";
  elements.localAiPanel.className = "local-ai-panel warning";
  elements.localAiTitle.textContent = "Loading the Balanced model…";
  elements.localAiDetail.textContent = "This removes most of the delay from the first local-AI answer.";
  try {
    const response = await request("/ollama/warmup", { method: "POST" });
    setStatus(elements.chatStatus, response.message || "Local AI is ready.");
    await loadHealth();
  } catch (error) {
    setStatus(elements.chatStatus, error.message, true);
    await loadHealth();
  }
});

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

Promise.all([loadHealth(), loadDocuments(), loadQualitySummary()]);
window.setInterval(loadHealth, 30000);
