"use strict";

const state = {
  files: [],
  jobs: [],
  activeJobId: null,
  previewMode: "result",
  pollTimer: null,
  uploading: false,
  serviceReady: false,
};

const els = {};
const statusLabels = {
  queued: "等待中",
  running: "处理中",
  completed: "已完成",
  failed: "失败",
};
const kindLabels = { group: "组", text: "字", shape: "形", pixel: "图" };
const acceptedTypes = new Map([
  ["png", "image/png"],
  ["jpg", "image/jpeg"],
  ["jpeg", "image/jpeg"],
  ["webp", "image/webp"],
]);
const maxFiles = 10;
const maxFileBytes = 20 * 1024 * 1024;
const maxBatchBytes = 80 * 1024 * 1024;

document.addEventListener("DOMContentLoaded", () => {
  ["health", "healthText", "healthRetry", "shutdownButton", "uploadForm", "dropZone", "fileInput", "selectedFiles",
    "formMessage", "submitButton", "refreshJobs", "jobCount", "jobList", "resultPanel", "liveStatus"]
    .forEach((id) => { els[id] = document.getElementById(id); });
  bindEvents();
  refreshHealth();
  refreshJobs({ announce: false });
});

function bindEvents() {
  els.healthRetry.addEventListener("click", refreshHealth);
  els.shutdownButton.addEventListener("click", shutdownApp);
  els.refreshJobs.addEventListener("click", () => refreshJobs({ announce: true }));
  els.uploadForm.addEventListener("submit", submitJobs);
  els.fileInput.addEventListener("change", () => selectFiles([...els.fileInput.files]));
  els.dropZone.addEventListener("click", () => els.fileInput.click());
  els.dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      els.fileInput.click();
    }
  });
  ["dragenter", "dragover"].forEach((name) => els.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropZone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach((name) => els.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropZone.classList.remove("is-dragging");
  }));
  els.dropZone.addEventListener("drop", (event) => selectFiles([...event.dataTransfer.files]));
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  let data = null;
  try { data = await response.json(); } catch (_) { /* handled below */ }
  if (!response.ok) {
    const detail = data && (data.detail || data.message);
    throw new Error(typeof detail === "string" ? detail : `请求失败（${response.status}）`);
  }
  if (!data || typeof data !== "object") throw new Error("服务返回了无法读取的数据");
  return data;
}

async function refreshHealth() {
  els.health.className = "health";
  els.healthText.textContent = "正在检查本机服务…";
  els.healthRetry.classList.add("is-hidden");
  try {
    const health = await fetchJson("/api/health");
    document.querySelector(".local-badge").textContent = `开源本机版 ${health.version}`;
    state.serviceReady = health.ready === true;
    els.health.classList.add(state.serviceReady ? "ready" : "error");
    const unreadyModels = Array.isArray(health.models) ? health.models.filter((model) => !model.ready).map((model) => model.name) : [];
    if (state.serviceReady) {
      els.healthText.textContent = health.message || "本机服务已就绪";
    } else {
      els.healthText.textContent = health.message || (unreadyModels.length ? `模型未就绪：${unreadyModels.join("、")}` : "本机服务未就绪");
      els.healthRetry.classList.remove("is-hidden");
    }
  } catch (error) {
    state.serviceReady = false;
    els.health.classList.add("error");
    els.healthText.textContent = "无法连接本机服务";
    els.healthRetry.classList.remove("is-hidden");
  }
  updateSubmitState();
}

function selectFiles(files) {
  els.formMessage.textContent = "";
  const valid = [];
  const rejected = [];
  files.slice(0, maxFiles).forEach((file) => {
    const extension = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    const expectedType = acceptedTypes.get(extension);
    if (!expectedType || file.type.toLowerCase() !== expectedType) {
      rejected.push(`${file.name || "未命名文件"}（仅支持 PNG、JPG、WebP，扩展名与格式须一致）`);
    } else if (file.size <= 0) {
      rejected.push(`${file.name}（文件为空）`);
    } else if (file.size > maxFileBytes) {
      rejected.push(`${file.name}（超过 20 MB）`);
    } else {
      valid.push(file);
    }
  });
  if (files.length > maxFiles) rejected.push(`另有 ${files.length - maxFiles} 张（每批最多 10 张）`);
  while (valid.reduce((total, file) => total + file.size, 0) > maxBatchBytes) {
    const removed = valid.pop();
    rejected.push(`${removed.name}（加入后整批超过 80 MB）`);
  }
  state.files = valid;
  if (rejected.length) els.formMessage.textContent = `已忽略：${rejected.join("；")}`;
  renderSelectedFiles();
  updateSubmitState();
}

function renderSelectedFiles() {
  els.selectedFiles.replaceChildren();
  els.selectedFiles.classList.toggle("is-hidden", state.files.length === 0);
  state.files.forEach((file, index) => {
    const row = element("div", "file-row");
    const info = element("div");
    info.append(element("strong", "", file.name), element("small", "", formatBytes(file.size)));
    const remove = element("button", "", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", `移除 ${file.name}`);
    remove.addEventListener("click", () => {
      state.files.splice(index, 1);
      els.fileInput.value = "";
      renderSelectedFiles();
      updateSubmitState();
    });
    row.append(info, remove);
    els.selectedFiles.append(row);
  });
}

function updateSubmitState() {
  els.submitButton.disabled = state.files.length === 0 || state.uploading || !state.serviceReady;
}

async function submitJobs(event) {
  event.preventDefault();
  if (!state.files.length) {
    els.formMessage.textContent = "请先选择至少一张图片。";
    return;
  }
  if (!state.serviceReady) {
    els.formMessage.textContent = "本机服务尚未就绪，请检查上方状态后重试。";
    return;
  }
  state.uploading = true;
  els.formMessage.textContent = "";
  els.submitButton.classList.add("is-loading");
  els.submitButton.querySelector("span").textContent = "正在提交";
  updateSubmitState();

  const formData = new FormData();
  state.files.forEach((file) => formData.append("files", file, file.name));
  formData.append("scale", new FormData(els.uploadForm).get("scale") || "3");
  formData.append("crop_ui", String(els.uploadForm.elements.crop_ui.checked));
  formData.append("split_light", String(els.uploadForm.elements.split_light.checked));

  try {
    const data = await fetchJson("/api/jobs", { method: "POST", body: formData });
    if (!Array.isArray(data.jobs) || data.jobs.length === 0) throw new Error("服务没有返回已创建的任务");
    mergeJobs(data.jobs, true);
    state.activeJobId = data.jobs[0].id;
    state.previewMode = "result";
    state.files = [];
    els.fileInput.value = "";
    renderSelectedFiles();
    renderAll();
    announce(`已创建 ${data.jobs.length} 个任务`);
    schedulePolling();
  } catch (error) {
    els.formMessage.textContent = `提交失败：${friendlyError(error)}`;
  } finally {
    state.uploading = false;
    els.submitButton.classList.remove("is-loading");
    els.submitButton.querySelector("span").textContent = "开始分层";
    updateSubmitState();
  }
}

async function shutdownApp() {
  const hasPendingJobs = state.jobs.some((job) => job.status === "queued" || job.status === "running");
  if (hasPendingJobs || state.uploading) {
    els.healthText.textContent = "还有图片正在处理，请等任务完成后再退出。";
    announce(els.healthText.textContent);
    return;
  }

  els.shutdownButton.disabled = true;
  els.health.className = "health";
  els.healthText.textContent = "正在退出软件…";
  els.healthRetry.classList.add("is-hidden");
  announce("正在退出软件");
  try {
    const response = await fetch("/api/shutdown", { method: "POST" });
    let data = null;
    try { data = await response.json(); } catch (_) { /* response text is optional */ }
    if (!response.ok) {
      const detail = data && (data.detail || data.message);
      throw new Error(typeof detail === "string" ? detail : `退出请求失败（${response.status}）`);
    }
    state.serviceReady = false;
    els.health.classList.add("ready");
    const message = data && typeof data.message === "string" ? data.message : "退出请求已接受，可以关闭此页面";
    els.healthText.textContent = message;
    announce(message);
    updateSubmitState();
  } catch (error) {
    els.health.classList.add("error");
    els.healthText.textContent = `退出失败：${friendlyError(error)}`;
    els.shutdownButton.disabled = false;
    announce(els.healthText.textContent);
  }
}

async function refreshJobs({ announce: shouldAnnounce = false } = {}) {
  els.refreshJobs.disabled = true;
  try {
    const data = await fetchJson("/api/jobs");
    if (!Array.isArray(data.jobs)) throw new Error("任务列表格式不正确");
    state.jobs = data.jobs;
    if (state.activeJobId && !state.jobs.some((job) => job.id === state.activeJobId)) state.activeJobId = null;
    if (!state.activeJobId && state.jobs.length) state.activeJobId = state.jobs[0].id;
    renderAll();
    if (shouldAnnounce) announce(`任务列表已刷新，共 ${state.jobs.length} 项`);
  } catch (error) {
    if (!state.jobs.length) renderJobsError(error);
    if (shouldAnnounce) announce(`刷新失败：${friendlyError(error)}`);
  } finally {
    els.refreshJobs.disabled = false;
    schedulePolling();
  }
}

function mergeJobs(incoming, prepend = false) {
  const byId = new Map(state.jobs.map((job) => [job.id, job]));
  incoming.forEach((job) => byId.set(job.id, job));
  if (prepend) {
    const incomingIds = new Set(incoming.map((job) => job.id));
    state.jobs = [...incoming, ...state.jobs.filter((job) => !incomingIds.has(job.id))];
  } else {
    state.jobs = state.jobs.map((job) => byId.get(job.id));
  }
}

function schedulePolling() {
  clearTimeout(state.pollTimer);
  const pending = state.jobs.filter((job) => job.status === "queued" || job.status === "running");
  if (!pending.length) return;
  state.pollTimer = setTimeout(pollPendingJobs, 2000);
}

async function pollPendingJobs() {
  const pendingIds = state.jobs
    .filter((job) => job.status === "queued" || job.status === "running")
    .map((job) => job.id);
  if (!pendingIds.length) return;
  const results = await Promise.allSettled(pendingIds.map((id) => fetchJson(`/api/jobs/${encodeURIComponent(id)}`)));
  const updates = [];
  results.forEach((result) => { if (result.status === "fulfilled") updates.push(result.value); });
  if (updates.length < results.length) {
    els.health.classList.add("error");
    els.healthText.textContent = "暂时无法读取处理进度，正在重试…";
  } else if (els.healthText.textContent === "暂时无法读取处理进度，正在重试…") {
    els.health.classList.remove("error");
    els.healthText.textContent = "本机服务已连接。";
  }
  if (updates.length) {
    const previous = new Map(state.jobs.map((job) => [job.id, job.status]));
    mergeJobs(updates);
    renderAll();
    updates.forEach((job) => {
      if (previous.get(job.id) !== job.status && (job.status === "completed" || job.status === "failed")) {
        announce(`${job.name}：${statusLabels[job.status]}`);
      }
    });
  }
  schedulePolling();
}

function renderAll() {
  renderJobList();
  renderResult();
}

function renderJobList() {
  els.jobList.replaceChildren();
  els.jobCount.textContent = String(state.jobs.length);
  if (!state.jobs.length) {
    const empty = element("div", "empty-state compact");
    empty.append(element("span"), element("p", "", "还没有任务"));
    els.jobList.append(empty);
    return;
  }
  state.jobs.forEach((job) => {
    const button = element("button", `job-card${job.id === state.activeJobId ? " is-active" : ""}`);
    button.type = "button";
    button.setAttribute("aria-pressed", String(job.id === state.activeJobId));
    const top = element("div", "job-card-top");
    top.append(element("span", "job-name", job.name || "未命名图片"), element("span", `status-tag ${job.status}`, statusLabels[job.status] || job.status || "未知"));
    const stage = element("div", "job-stage", job.stage || "尚未开始");
    button.append(top, stage);
    if (job.status === "queued" || job.status === "running") {
      const track = element("div", "progress-track");
      track.setAttribute("aria-label", `进度 ${safeProgress(job.progress)}%`);
      const fill = element("span");
      fill.style.width = `${safeProgress(job.progress)}%`;
      track.append(fill);
      button.append(track);
    }
    const meta = element("div", "job-card-meta");
    meta.append(element("span", "", formatDate(job.created_at)), element("span", "", formatElapsed(job.elapsed)));
    button.append(meta);
    button.addEventListener("click", () => {
      state.activeJobId = job.id;
      state.previewMode = "result";
      renderAll();
    });
    els.jobList.append(button);
  });
}

function renderJobsError(error) {
  els.jobCount.textContent = "!";
  els.jobList.replaceChildren();
  const empty = element("div", "empty-state compact");
  empty.append(element("span"), element("p", "", `读取失败：${friendlyError(error)}`));
  els.jobList.append(empty);
}

function renderResult() {
  const job = state.jobs.find((item) => item.id === state.activeJobId);
  els.resultPanel.replaceChildren();
  if (!job) {
    const empty = element("div", "empty-state result-empty");
    const art = element("div", "empty-art");
    art.setAttribute("aria-hidden", "true");
    art.append(element("span"), element("span"), element("span"));
    empty.append(art, element("p", "", "选择一项任务"), element("small", "", "处理状态、预览和可编辑图层会出现在这里"));
    els.resultPanel.append(empty);
    return;
  }
  els.resultPanel.append(buildResultHeader(job));
  if (job.status === "completed" && job.result) renderCompleted(job);
  else if (job.status === "failed") renderFailed(job);
  else renderPending(job);
}

function buildResultHeader(job) {
  const header = element("header", "result-header");
  const title = element("div", "result-title");
  title.append(element("h3", "", job.name || "未命名图片"), element("p", "", `${formatDate(job.created_at)} · ${formatElapsed(job.elapsed)}`));
  const status = element("div", `result-status ${job.status}`);
  status.append(element("i"), element("span", "", statusLabels[job.status] || job.status || "未知"));
  header.append(title, status);
  return header;
}

function renderPending(job) {
  const view = element("div", "pending-view");
  const number = element("div", "pending-number");
  number.append(document.createTextNode(String(safeProgress(job.progress))), element("small", "", "%"));
  const track = element("div", "large-progress");
  const fill = element("span");
  fill.style.width = `${safeProgress(job.progress)}%`;
  track.append(fill);
  view.append(number, track, element("h4", "", job.stage || (job.status === "queued" ? "正在等待处理" : "正在处理")), element("p", "", "这里显示的是处理服务返回的真实阶段与进度。任务可在后台继续运行。"));
  els.resultPanel.append(view);
}

function renderFailed(job) {
  const view = element("div", "failed-view");
  view.append(element("h4", "", job.stage || "任务处理失败"), element("p", "", job.error || "服务未提供详细错误。请重新选择图片后提交。"));
  els.resultPanel.append(view);
}

function renderCompleted(job) {
  const result = job.result;
  const files = result.files || {};
  const reviewRequired = Array.isArray(result.review_required) ? result.review_required : [];
  const preservedCount = Number.isFinite(result.text_preserved_count) ? result.text_preserved_count : 0;
  if (reviewRequired.length || preservedCount) {
    const note = element('div', 'quality-notice');
    const hasTextGroups = Number.isFinite(result.text_region_count);
    note.append(element('strong', '', hasTextGroups
      ? `${result.text_region_count} 处透明原字形，${preservedCount} 处保真原区域（含背景、需精修）`
      : `${reviewRequired.length} 处文字区域未拆分，需确认`));
    note.append(element('p', '', hasTextGroups
      ? '透明原字形可移动、叠加颜色及新增描边；保真原区域仍含背景，不能直接改字，需精修后才能只调整字形。改字时先关闭对应原字形及边缘阴影，或关闭保真原区域，再打开隐藏的“可改字替代”。替代字体和字效为近似重建，包装小字请先核对内容。'
      : '这些区域保留了原字形和画面，仍在像素层中，不能直接改字或独立调整其光影。'));
    if (preservedCount) note.append(element('p', '', '含背景区域移动时会带动背景；对整个区域换色或新增描边也会影响背景，尚未达到独立透明字形的编辑效果。'));
    const list = element('ul');
    reviewRequired.forEach((item) => list.append(element('li', '', `${item.text}：${item.reason}`)));
    const details = element('details');
    details.append(element('summary', '', hasTextGroups ? '查看需要精修或核对的文字' : '查看未拆分的区域'), list);
    note.append(details);els.resultPanel.append(note);
  }
  const previewUrl = safeArtifactUrl(files.preview);
  const originalUrl = safeArtifactUrl(files.original);
  const toolbar = element("div", "preview-toolbar");
  const switcher = element("div", "view-switch");
  const originalButton = element("button", state.previewMode === "original" ? "is-active" : "", "原图");
  const resultButton = element("button", state.previewMode === "result" ? "is-active" : "", "合成结果");
  originalButton.type = resultButton.type = "button";
  originalButton.setAttribute("aria-pressed", String(state.previewMode === "original"));
  resultButton.setAttribute("aria-pressed", String(state.previewMode === "result"));
  originalButton.addEventListener("click", () => { state.previewMode = "original"; renderResult(); });
  resultButton.addEventListener("click", () => { state.previewMode = "result"; renderResult(); });
  switcher.append(originalButton, resultButton);
  toolbar.append(switcher, element("span", "preview-size", `${numberOrDash(result.width)} × ${numberOrDash(result.height)} px`));

  const stage = element("div", "preview-stage");
  const imageUrl = state.previewMode === "original" ? originalUrl : previewUrl;
  if (imageUrl) {
    const img = document.createElement("img");
    img.src = imageUrl;
    img.alt = state.previewMode === "original" ? `${job.name} 原图` : `${job.name} PSD 合成预览`;
    img.addEventListener("error", () => {
      stage.replaceChildren(element("p", "preview-error", "预览图片暂时无法载入，仍可尝试下载结果文件。"));
    });
    stage.append(img);
  } else {
    stage.append(element("p", "preview-error", "服务未返回此预览文件。"));
  }
  els.resultPanel.append(toolbar, stage, buildSummary(result), buildDetails(result));
}

function buildSummary(result) {
  const summary = element("div", "result-summary");
  const statsBlock = element("section", "summary-block");
  statsBlock.append(element("h4", "", "重建概览"));
  const stats = element("div", "stat-grid");
  [[result.layer_count, "图层"], [result.text_count, "原生文字"], [result.pixel_count, "像素层"], [result.shape_count, "形状层"]].forEach(([value, label]) => {
    const item = element("div", "stat");
    item.append(element("strong", "", numberOrDash(value)), element("span", "", label));
    stats.append(item);
  });
  statsBlock.append(stats);
  if (Number.isFinite(result.detected_text_count)) {
    const preservedCount = Number.isFinite(result.text_preserved_count) ? result.text_preserved_count : 0;
    const sharedCount = Number.isFinite(result.shared_text_count) ? result.shared_text_count : 0;
    statsBlock.append(element('p', '', `识别 ${result.detected_text_count} 处文字 · ${result.text_count} 处直接可改字 · ${result.text_region_count} 处透明原字形 · ${preservedCount} 处保真原区域（含背景、需精修） · ${sharedCount} 处共享字形`));
    statsBlock.append(element('p', '', `另附 ${result.text_alternative_count} 个默认隐藏的近似改字替代层，未计入直接可改字数量。`));
    const textList = element('details', 'text-coverage');
    const list = element('ul');
    const modeLabels = {
      native:'原生可改字', shared:'字形见相邻组，附独立改字替代',
      appearance:'透明原字形 + 近似改字替代', preserved:'保真原区域（含背景、需精修）+ 近似改字替代',
    };
    (result.text_audit || []).forEach((item) => list.append(element('li', '', `${item.text}：${item.contains_background ? modeLabels.preserved : modeLabels[item.mode] || '处理方式待确认'}`)));
    textList.append(element('summary', '', '查看全部识别文字及处理方式'), list);
    statsBlock.append(textList);
  }

  const downloadBlock = element("section", "summary-block");
  downloadBlock.append(element("h4", "", "下载文件"));
  const downloads = element("div", "downloads");
  const downloadItems = [
    [result.files && result.files.psd, "PSD"],
    [result.files && result.files.zip, "完整包"],
    [result.files && result.files.report, "说明"],
    [result.files && result.files.preview, "预览"],
    [result.files && result.files.original, "原图"],
  ];
  downloadItems.forEach(([url, label]) => {
    const safeUrl = safeArtifactUrl(url);
    if (!safeUrl) return;
    const link = element("a", "download-link", `${label} ↓`);
    link.href = safeUrl;
    link.setAttribute("download", "");
    downloads.append(link);
  });
  if (!downloads.children.length) downloads.append(element("span", "", "服务未返回可下载文件"));
  downloadBlock.append(downloads);
  summary.append(statsBlock, downloadBlock);
  return summary;
}

function buildDetails(result) {
  const container = element("div", "details-row");
  const layerDetails = document.createElement("details");
  layerDetails.open = false;
  const layerSummary = element("summary");
  layerSummary.append(element("span", "", "真实图层结构"), element("small", "", `${Array.isArray(result.layers) ? result.layers.length : 0} 个顶层项`));
  layerDetails.append(layerSummary);
  const tree = element("ul", "layer-tree");
  if (Array.isArray(result.layers) && result.layers.length) result.layers.forEach((layer) => tree.append(buildLayerNode(layer)));
  else tree.append(element("li", "layer-item", "未返回图层摘要"));
  layerDetails.append(tree);
  container.append(layerDetails);

  const warnings = Array.isArray(result.warnings) ? result.warnings : [];
  if (warnings.length || result.enhancement) {
    const noteDetails = document.createElement("details");
    const noteSummary = element("summary");
    noteSummary.append(element("span", "", "重建说明与提醒"), element("small", "", warnings.length ? `${warnings.length} 条` : ""));
    noteDetails.append(noteSummary);
    if (result.enhancement) noteDetails.append(element("p", "enhancement", result.enhancement));
    if (warnings.length) {
      const list = element("ul", "warning-list");
      warnings.forEach((warning) => list.append(element("li", "", String(warning))));
      noteDetails.append(list);
    }
    container.append(noteDetails);
  }
  return container;
}

function buildLayerNode(layer) {
  const li = document.createElement("li");
  const hasChildren = Array.isArray(layer.children) && layer.children.length > 0;
  const row = element(hasChildren ? "summary" : "div", "layer-item");
  row.append(element("span", "layer-icon", kindLabels[layer.kind] || "·"), element("strong", "", layer.name || "未命名图层"));
  if (hasChildren) {
    const group = document.createElement("details");
    group.open = true;
    group.append(row);
    const children = document.createElement("ul");
    layer.children.forEach((child) => children.append(buildLayerNode(child)));
    group.append(children);
    li.append(group);
  } else {
    li.append(row);
  }
  return li;
}

function safeArtifactUrl(value) {
  if (typeof value !== "string" || !value) return null;
  try {
    const url = new URL(value, window.location.origin);
    return url.origin === window.location.origin && (url.protocol === "http:" || url.protocol === "https:") ? url.href : null;
  } catch (_) { return null; }
}

function element(tag, className = "", textContent) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (textContent !== undefined) node.textContent = String(textContent);
  return node;
}

function safeProgress(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.max(0, Math.min(100, Math.round(numeric))) : 0;
}

function numberOrDash(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toLocaleString("zh-CN") : "—";
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "大小未知";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value) {
  if (!value) return "时间未知";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date);
}

function formatElapsed(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds < 0) return "耗时未知";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} 秒`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} 分 ${Math.round(seconds % 60)} 秒`;
}

function friendlyError(error) {
  if (error instanceof TypeError) return "无法连接本机服务，请确认程序仍在运行";
  return error && error.message ? error.message : "发生未知错误";
}

function announce(message) {
  els.liveStatus.textContent = "";
  window.setTimeout(() => { els.liveStatus.textContent = message; }, 20);
}
