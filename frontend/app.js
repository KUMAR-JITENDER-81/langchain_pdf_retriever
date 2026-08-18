const API_BASE = window.API_BASE || "http://127.0.0.1:8000";
const API_TOKEN = window.API_TOKEN || "";
const documentsElement = document.querySelector("#documents");
const uploadStatus = document.querySelector("#upload-status");
const chatStatus = document.querySelector("#chat-status");
const answerElement = document.querySelector("#answer");

function setStatus(element, message, error = false) {
  element.textContent = message;
  element.classList.toggle("error", error);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;",
  })[character]);
}

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (API_TOKEN) headers.set("Authorization", `Bearer ${API_TOKEN}`);
  options.headers = headers;
  const response = await fetch(`${API_BASE}${path}`, options);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || body.message || "Request failed");
  }
  return body;
}

async function loadDocuments() {
  const response = await request("/documents/");
  const documents = response.data.documents;
  documentsElement.classList.toggle("empty", documents.length === 0);
  documentsElement.innerHTML = documents.length
    ? documents.map((document) => `
        <div class="document-row">
          <div>
            <strong>${escapeHtml(document.stored_filename)}</strong>
            <small>${escapeHtml(document.size_bytes)} bytes · ${escapeHtml(document.document_id)}</small>
          </div>
          <div class="actions">
            <button data-index="${document.document_id}" class="secondary">Index</button>
            <button data-delete="${document.document_id}" class="danger">Delete</button>
          </div>
        </div>`).join("")
    : "No documents uploaded yet.";
}

document.querySelector("#upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.querySelector("#pdf-file").files[0];
  const formData = new FormData();
  formData.append("file", file);
  setStatus(uploadStatus, "Uploading...");
  try {
    await request("/upload/", { method: "POST", body: formData });
    setStatus(uploadStatus, "PDF uploaded successfully.");
    event.target.reset();
    await loadDocuments();
  } catch (error) {
    setStatus(uploadStatus, error.message, true);
  }
});

documentsElement.addEventListener("click", async (event) => {
  const indexId = event.target.dataset.index;
  const deleteId = event.target.dataset.delete;
  try {
    if (indexId) {
      setStatus(uploadStatus, "Indexing document...");
      await request(`/documents/${indexId}/index`, { method: "POST" });
      setStatus(uploadStatus, "Document indexed successfully.");
    }
    if (deleteId) {
      await request(`/documents/${deleteId}`, { method: "DELETE" });
      await loadDocuments();
      setStatus(uploadStatus, "Document deleted.");
    }
  } catch (error) {
    setStatus(uploadStatus, error.message, true);
  }
});

document.querySelector("#refresh-documents").addEventListener("click", loadDocuments);

document.querySelector("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(chatStatus, "Searching documents and generating an answer...");
  answerElement.classList.add("hidden");
  try {
    const response = await request("/chat/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: document.querySelector("#question").value, k: 4 }),
    });
    answerElement.innerHTML = `<p>${escapeHtml(response.data.answer)}</p><small>Sources: ${escapeHtml(response.data.sources.map((source) => `Page ${source.page}`).join(", ") || "None")}</small>`;
    answerElement.classList.remove("hidden");
    setStatus(chatStatus, "");
  } catch (error) {
    setStatus(chatStatus, error.message, true);
  }
});

loadDocuments().catch((error) => setStatus(uploadStatus, error.message, true));
