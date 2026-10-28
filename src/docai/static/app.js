"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(String(child ?? "")));
  return node;
};

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}

const display = (value) => (value === null || value === undefined ? "" : typeof value === "object" ? JSON.stringify(value, null, 1) : String(value));

async function refreshLists() {
  const [queue, docs] = await Promise.all([api("/api/review"), api("/api/documents")]);
  $("#queue-count").textContent = `(${queue.length})`;
  $("#queue").replaceChildren(...queue.map((f) =>
    el("li", { onclick: () => showDocument(f.document_id) }, `${f.name} — ${f.confidence.toFixed(2)}`, el("small", {}, f.filename))));
  $("#documents").replaceChildren(...docs.map((d) =>
    el("li", { onclick: () => showDocument(d.id) }, `#${d.id} ${d.filename}`, el("small", {}, `${d.method} · ${d.pending || 0} to review`))));
}

async function resolve(field, action, input) {
  let value = null;
  if (action === "correct") {
    value = input.value.trim();
    if (field.name === "line_items") {
      try { value = JSON.parse(value || "[]"); } catch { alert("line_items must be JSON"); return; }
    }
  }
  try {
    await api(`/api/fields/${field.id}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, value, reviewer: $("#reviewer").value || "reviewer" }),
    });
    await showDocument(field.document_id);
    await refreshLists();
  } catch (err) { alert(err.message); }
}

function fieldRow(field) {
  const editing = field.status === "needs_review";
  const input = field.name === "line_items" ? el("textarea", {}) : el("input", {});
  input.value = display(field.value);
  const valueCell = editing
    ? el("td", {}, input, el("div", { class: "actions" },
        el("button", { onclick: () => resolve(field, "correct", input) }, "Save correction"),
        el("button", { class: "secondary", onclick: () => resolve(field, "confirm", input) }, "Value is right")))
    : el("td", {}, field.name === "line_items" ? `${(field.value || []).length} lines` : display(field.value));
  return el("tr", { class: field.status },
    el("th", {}, field.name),
    valueCell,
    el("td", { class: "conf", title: JSON.stringify(field.signals) }, field.confidence.toFixed(2)),
    el("td", { class: "status" }, field.status.replace("_", " "),
      editing ? el("div", { class: "issues" }, field.issues.join("; ")) : ""));
}

async function showDocument(id) {
  const doc = await api(`/api/documents/${id}`);
  const pages = Array.from({ length: doc.pages }, (_, i) => el("img", { src: `/api/documents/${id}/pages/${i}`, alt: `page ${i + 1}` }));
  const tokens = doc.usage.prompt_tokens + doc.usage.completion_tokens;
  $("#detail").replaceChildren(
    el("div", { class: "meta" }, `#${doc.id} ${doc.filename} · ${doc.method} · ${doc.language} · ${Math.round(doc.timings_ms.total)} ms · ${tokens} tokens`),
    el("div", { class: "doc" },
      el("div", {}, ...pages),
      el("div", {},
        el("table", {}, el("tr", {}, el("th", {}, "field"), el("th", {}, "value"), el("th", {}, "conf."), el("th", {}, "status")),
          ...doc.fields.map(fieldRow)),
        doc.issues.length ? el("p", { class: "issues" }, "Validation: " + doc.issues.map((i) => i.message || i.code).join("; ")) : "")));
}

$("#upload").addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = $("#upload-status");
  status.textContent = "processing…";
  try {
    const doc = await api("/api/documents", { method: "POST", body: new FormData(event.target) });
    status.textContent = "";
    await refreshLists();
    await showDocument(doc.id);
  } catch (err) { status.textContent = err.message; }
});

refreshLists().catch((err) => { $("#queue").replaceChildren(el("li", {}, err.message)); });
