const canvas = document.querySelector("#scene");
const ctx = canvas.getContext("2d");
const annotatedFrame = new Image();

const els = {
  annotatedFrame,
  alertList: document.querySelector("#alertList"),
  apiBase: document.querySelector("#apiBase"),
  backendDetail: document.querySelector("#backendDetail"),
  canvasWrap: document.querySelector(".canvas-wrap"),
  clock: document.querySelector("#clock"),
  confidence: document.querySelector("#confidence"),
  eventRows: document.querySelector("#eventRows"),
  fileImage: document.querySelector("#fileImage"),
  fileVideo: document.querySelector("#fileVideo"),
  fps: document.querySelector("#fps"),
  imageInput: document.querySelector("#imageInput"),
  latency: document.querySelector("#latency"),
  liveChip: document.querySelector("#liveChip"),
  modelSelect: document.querySelector("#modelSelect"),
  refreshHealth: document.querySelector("#refreshHealth"),
  resultTime: document.querySelector("#resultTime"),
  riskBadge: document.querySelector("#riskBadge"),
  saveApi: document.querySelector("#saveApi"),
  sourceHint: document.querySelector("#sourceHint"),
  sourceMode: document.querySelector("#sourceMode"),
  sourceName: document.querySelector("#sourceName"),
  sourceTitle: document.querySelector("#sourceTitle"),
  statusStrip: document.querySelector("#statusStrip"),
  streamImage: document.querySelector("#streamImage"),
  systemStatus: document.querySelector("#systemStatus"),
  targetCount: document.querySelector("#targetCount"),
  toggleRun: document.querySelector("#toggleRun"),
  useAtlasStream: document.querySelector("#useAtlasStream"),
  useImage: document.querySelector("#useImage"),
  useVideo: document.querySelector("#useVideo"),
  videoControls: document.querySelector("#videoControls"),
  videoInput: document.querySelector("#videoInput"),
  videoSeek: document.querySelector("#videoSeek"),
  videoTime: document.querySelector("#videoTime"),
};

const CLASS_META = {
  crane: { zh: "吊车闯入", risk: "high", color: "#d33b31" },
  excavator: { zh: "挖掘机靠近", risk: "high", color: "#d33b31" },
  foreign_object: { zh: "导线异物", risk: "medium", color: "#b87912" },
};

const DEFAULT_API =
  new URLSearchParams(window.location.search).get("api")
  || localStorage.getItem("atlas_api_base")
  || "http://192.168.137.100:8000";

const DEFAULT_MODEL_ID =
  new URLSearchParams(window.location.search).get("model_id")
  || localStorage.getItem("atlas_model_id")
  || "";

const state = {
  apiBase: DEFAULT_API,
  modelId: DEFAULT_MODEL_ID,
  models: [],
  connected: false,
  connecting: false,
  processing: false,
  modelReady: false,
  running: true,
  mode: "stream",
  sourceLabel: "atlas-camera://0",
  sourceHint: "等待 Atlas 后端提供真实检测结果",
  detections: [],
  eventHistory: [],
  eventSource: null,
  streamResult: null,
  lastResult: null,
  lastHealth: null,
  frameCount: 0,
  fps: 0,
  lastFpsTime: performance.now(),
  imageAsset: {
    name: "",
    url: "",
    annotatedUrl: "",
    result: null,
  },
  videoAsset: {
    name: "",
    url: "",
    result: null,
    events: [],
    currentTime: 0,
    duration: 0,
    currentAnnotatedImage: "",
    streamController: null,
    streamFrameUrl: "",
    streaming: false,
  },
  videoEventIndex: -1,
  seekDragging: false,
};

els.apiBase.value = state.apiBase;
els.modelSelect.value = state.modelId;

function apiUrl(path) {
  if (/^https?:\/\//i.test(path)) return path;
  return `${state.apiBase.replace(/\/$/, "")}${path}`;
}

function apiUrlWithModel(path, extraParams = {}) {
  const url = new URL(apiUrl(path));
  if (state.modelId) {
    url.searchParams.set("model_id", state.modelId);
  }
  Object.entries(extraParams).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  });
  return url.toString();
}

function appendModelId(form) {
  if (state.modelId) {
    form.set("model_id", state.modelId);
  }
  return form;
}

function setConnection(status, detail = "", preserveDetail = false) {
  state.connected = status === "online";
  state.connecting = status === "connecting";
  els.statusStrip.className = `status-strip ${status === "online" ? "" : status}`;
  els.systemStatus.textContent =
    status === "online"
      ? "Atlas 后端已连接"
      : status === "connecting"
        ? "正在连接 Atlas 后端"
        : "未连接开发板后端";

  if (!preserveDetail) {
    els.backendDetail.textContent = detail || (state.connected ? "后端在线" : "等待后端响应");
  }

  els.liveChip.className = `live-chip ${state.connected ? "" : "offline"}`;
  els.liveChip.textContent = state.connected ? (state.mode === "stream" ? "LIVE" : "RESULT") : "OFFLINE";
}

function setSource(mode, label, title, hint) {
  state.mode = mode;
  state.sourceLabel = label;
  state.sourceHint = hint;
  els.sourceTitle.textContent = title;
  els.sourceName.textContent = label;
  els.sourceHint.textContent = hint;
  els.sourceMode.textContent = label;

  [els.useAtlasStream, els.useImage, els.useVideo].forEach((button) => button.classList.remove("active"));
  if (mode === "stream") els.useAtlasStream.classList.add("active");
  if (mode === "image") els.useImage.classList.add("active");
  if (mode === "video") els.useVideo.classList.add("active");

  updateVideoControlsVisibility();
  updateCanvasAspect();
}

function setSourceHint(hint) {
  state.sourceHint = hint;
  els.sourceHint.textContent = hint;
}

function updateVideoControlsVisibility() {
  const showControls = state.mode === "video" && Boolean(state.videoAsset.url) && !state.videoAsset.streaming;
  els.videoControls.classList.toggle("hidden", !showControls);
}

function formatTime(totalSeconds) {
  const seconds = Math.max(0, Math.floor(totalSeconds || 0));
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}

function updateVideoTimelineUi() {
  const duration = Number.isFinite(els.fileVideo.duration) ? els.fileVideo.duration : state.videoAsset.duration || 0;
  const currentTime = Number.isFinite(els.fileVideo.currentTime) ? els.fileVideo.currentTime : state.videoAsset.currentTime || 0;
  state.videoAsset.duration = duration;
  state.videoAsset.currentTime = currentTime;

  if (duration > 0) {
    const value = Math.min(1000, Math.max(0, Math.round((currentTime / duration) * 1000)));
    if (!state.seekDragging) {
      els.videoSeek.value = String(value);
    }
  } else if (!state.seekDragging) {
    els.videoSeek.value = "0";
  }

  els.videoTime.textContent = `${formatTime(currentTime)} / ${formatTime(duration)}`;
}

function resetVideoTimeline() {
  state.videoAsset.events = [];
  state.videoEventIndex = -1;
  updateVideoTimelineUi();
}

function clearCurrentVisualResult() {
  state.detections = [];
  state.lastResult = null;
  els.resultTime.textContent = "--";
  renderMetrics();
  renderAlerts("等待后端检测结果");
}

function renderAlerts(message) {
  if (state.detections.length === 0) {
    els.alertList.innerHTML = `<div class="alert low"><strong>${escapeHtml(message)}</strong><small>来自后端的真实检测结果为空</small></div>`;
    return;
  }

  els.alertList.innerHTML = state.detections
    .map((det) => `<div class="alert ${det.risk}"><strong>${escapeHtml(det.zh)}</strong><small>${escapeHtml(det.label)} · ${(det.score * 100).toFixed(1)}% · 后端检测结果</small></div>`)
    .join("");
}

async function fetchJson(path, options = {}) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), options.timeout ?? 5000);

  try {
    const response = await fetch(apiUrl(path), {
      ...options,
      signal: controller.signal,
    });
    const text = await response.text();
    let data = {};
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { message: text };
      }
    }

    if (!response.ok) {
      throw new Error(data.error || data.message || `HTTP ${response.status}`);
    }

    return data;
  } finally {
    window.clearTimeout(timer);
  }
}

async function loadModels() {
  const data = await fetchJson(apiUrlWithModel("/api/models"), { timeout: 3500 });
  state.models = data.models || [];

  if (!state.modelId) {
    state.modelId = data.selected_model_id || data.default_model_id || "";
    localStorage.setItem("atlas_model_id", state.modelId);
  }

  if (state.modelId && !state.models.some((model) => model.id === state.modelId)) {
    state.modelId = data.default_model_id || state.models[0]?.id || "";
    localStorage.setItem("atlas_model_id", state.modelId);
  }

  renderModelSelect();
}

function renderModelSelect() {
  if (!els.modelSelect) return;

  if (!state.models.length) {
    els.modelSelect.innerHTML = '<option value="">Default model</option>';
    els.modelSelect.value = "";
    return;
  }

  els.modelSelect.innerHTML = state.models
    .map((model) => {
      const status = model.available ? "" : " (missing)";
      const label = `${model.name || model.id}${status}`;
      return `<option value="${escapeHtml(model.id)}">${escapeHtml(label)}</option>`;
    })
    .join("");
  els.modelSelect.value = state.modelId || state.models[0].id;
}

async function readMjpegStream(stream, onFrame) {
  const reader = stream.getReader();
  let buffer = new Uint8Array(0);

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer = concatBytes(buffer, value);

      while (true) {
        const start = indexOfBytes(buffer, [0xff, 0xd8]);
        if (start < 0) {
          if (buffer.length > 8192) buffer = buffer.slice(-8192);
          break;
        }

        const end = indexOfBytes(buffer, [0xff, 0xd9], start + 2);
        if (end < 0) {
          if (start > 8192) buffer = buffer.slice(start - 8192);
          break;
        }

        const metadata = parseMjpegMetadata(buffer.slice(0, start));
        const jpegBytes = buffer.slice(start, end + 2);
        await onFrame(jpegBytes, metadata);
        buffer = buffer.slice(end + 2);
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function concatBytes(left, right) {
  const result = new Uint8Array(left.length + right.length);
  result.set(left, 0);
  result.set(right, left.length);
  return result;
}

function indexOfBytes(buffer, pattern, from = 0) {
  for (let i = from; i <= buffer.length - pattern.length; i += 1) {
    let matched = true;
    for (let j = 0; j < pattern.length; j += 1) {
      if (buffer[i + j] !== pattern[j]) {
        matched = false;
        break;
      }
    }
    if (matched) return i;
  }
  return -1;
}

function parseMjpegMetadata(headerBytes) {
  const headerText = new TextDecoder().decode(headerBytes);
  const matches = [...headerText.matchAll(/X-Atlas-Result:\s*([A-Za-z0-9+/=]+)/gi)];
  if (!matches.length) return null;
  return parseBase64Json(matches[matches.length - 1][1]);
}

function parseBase64Json(value) {
  const binary = atob(value);
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
  return JSON.parse(new TextDecoder().decode(bytes));
}

async function checkHealth() {
  if (state.processing) return;

  if (!state.connected) setConnection("connecting", "请求 /api/health");
  else els.backendDetail.textContent = "刷新后端状态";

  try {
    const data = await fetchJson(apiUrlWithModel("/api/health"), { timeout: 3500 });
    state.lastHealth = data;
    state.modelReady = Boolean(data.model_loaded);
    if (data.model_id) {
      state.modelId = data.model_id;
      localStorage.setItem("atlas_model_id", state.modelId);
      renderModelSelect();
    }
    const camera = data.camera_source ?? "0";
    const detail = data.model_loaded
      ? `模型已加载 · 摄像头 ${camera}`
      : `后端在线 · 模型未加载：${data.error || "等待 OM 模型"}`;

    setConnection("online", detail, state.mode !== "stream" && Boolean(state.lastResult));

    if (state.mode === "stream") {
      setSource("stream", `atlas-camera://${camera}`, "Atlas 摄像头实时流", "开发板摄像头真实推理结果");
    }
  } catch (error) {
    state.modelReady = false;
    setConnection("offline", error.name === "AbortError" ? "后端无响应" : error.message);
    clearCurrentVisualResult();
    stopAtlasStream();
  }
}

function stopEvents() {
  if (!state.eventSource) return;
  state.eventSource.close();
  state.eventSource = null;
}

function startEvents() {
  stopEvents();

  try {
    state.eventSource = new EventSource(apiUrlWithModel("/api/events"));
  } catch (error) {
    els.backendDetail.textContent = `事件流启动失败：${error.message}`;
    return;
  }

  state.eventSource.addEventListener("result", (event) => {
    if (state.mode !== "stream") return;
    const payload = JSON.parse(event.data);
    applyResult(payload, { appendEvent: true, keepVideoTimeline: false, storeMode: "stream" });
  });

  state.eventSource.addEventListener("status", (event) => {
    if (state.mode !== "stream") return;
    const payload = JSON.parse(event.data);
    els.backendDetail.textContent = payload.message || "等待摄像头视频帧";
  });

  state.eventSource.onerror = () => {
    if (state.mode === "stream") {
      els.backendDetail.textContent = "实时事件流断开";
    }
  };
}

function stopAtlasStream() {
  els.streamImage.removeAttribute("src");
  els.streamImage.src = "";
  stopEvents();
}

function stopUploadedVideoStream() {
  if (state.videoAsset.streamController) {
    state.videoAsset.streamController.abort();
    state.videoAsset.streamController = null;
  }
  state.videoAsset.events.forEach((event) => {
    if (event.annotatedImage?.startsWith("blob:")) {
      URL.revokeObjectURL(event.annotatedImage);
    }
  });
  if (state.videoAsset.streamFrameUrl) {
    URL.revokeObjectURL(state.videoAsset.streamFrameUrl);
    state.videoAsset.streamFrameUrl = "";
  }
  state.videoAsset.events = [];
  state.videoAsset.streaming = false;
  state.videoAsset.currentAnnotatedImage = "";
  els.annotatedFrame.removeAttribute("src");
  updateVideoControlsVisibility();
}

function pauseStoredVideo() {
  els.fileVideo.pause();
}

function revokeAssetUrl(asset) {
  if (!asset.url) return;
  URL.revokeObjectURL(asset.url);
  asset.url = "";
}

function openImagePicker() {
  if (state.processing) return;
  els.imageInput.value = "";
  els.imageInput.click();
}

function openVideoPicker() {
  if (state.processing) return;
  els.videoInput.value = "";
  els.videoInput.click();
}

function showStoredImage() {
  if (!state.imageAsset.url) {
    openImagePicker();
    return;
  }

  stopAtlasStream();
  pauseStoredVideo();
  setSource("image", `upload-image://${state.imageAsset.name}`, state.imageAsset.name, "已加载图片检测结果");
  els.fileImage.src = state.imageAsset.result?.annotatedImage || state.imageAsset.url;

  if (state.imageAsset.result) {
    state.lastResult = state.imageAsset.result;
    state.detections = state.imageAsset.result.detections;
    els.resultTime.textContent = state.imageAsset.result.time;
    renderPanel();
  } else {
    clearCurrentVisualResult();
    setSourceHint("图片已保留，请重新检测");
  }
}

function showStoredVideo() {
  if (!state.videoAsset.url) {
    openVideoPicker();
    return;
  }

  stopAtlasStream();
  setSource("video", `upload-video://${state.videoAsset.name}`, state.videoAsset.name, "视频已保留，可自由拖动查看结果");
  els.fileVideo.src = state.videoAsset.url;
  els.fileVideo.currentTime = state.videoAsset.currentTime || 0;
  updateVideoControlsVisibility();

  if (state.videoAsset.result) {
    state.lastResult = state.videoAsset.result;
    state.detections = state.videoAsset.result.detections;
    els.resultTime.textContent = state.videoAsset.result.time;
    syncVideoResult(true);
    renderPanel();
  } else {
    clearCurrentVisualResult();
    setSourceHint("视频已保留，请重新检测");
  }
}

function startAtlasStream() {
  if (state.processing) return;

  stopUploadedVideoStream();
  stopAtlasStream();
  pauseStoredVideo();
  clearCurrentVisualResult();
  setSource("stream", "atlas-camera://0", "Atlas 摄像头实时流", "开发板摄像头真实推理结果");

  if (!state.connected || !state.modelReady) {
    clearCurrentVisualResult();
    setSourceHint(state.connected ? "模型未加载，实时流不可用" : "未连接开发板后端");
    return;
  }

  els.streamImage.src = apiUrlWithModel("/api/stream", { t: Date.now() });
  startEvents();
}

async function uploadMedia(file, kind) {
  if (!file) return;
  if (state.processing) return;

  stopUploadedVideoStream();
  stopAtlasStream();
  pauseStoredVideo();

  if (kind === "image") {
    revokeAssetUrl(state.imageAsset);
    state.imageAsset = { name: file.name, url: URL.createObjectURL(file), annotatedUrl: "", result: null };
    els.fileImage.src = state.imageAsset.url;
    setSource("image", `upload-image://${file.name}`, file.name, "等待后端图片检测结果");
  } else {
    revokeAssetUrl(state.videoAsset);
    state.videoAsset = {
      name: file.name,
      url: URL.createObjectURL(file),
      result: null,
      events: [],
      currentTime: 0,
      duration: 0,
      currentAnnotatedImage: "",
      streamController: null,
      streamFrameUrl: "",
      streaming: false,
    };
    els.fileVideo.src = state.videoAsset.url;
    els.fileVideo.currentTime = 0;
    setSource("video", `upload-video://${file.name}`, file.name, "等待后端视频检测结果");
    updateVideoControlsVisibility();
    updateVideoTimelineUi();
  }

  clearCurrentVisualResult();

  if (!state.connected || !state.modelReady) {
    setSourceHint(state.connected ? "模型未加载，无法检测上传文件" : "未连接开发板后端");
    return;
  }

  state.processing = true;
  els.liveChip.className = "live-chip processing";
  els.liveChip.textContent = "RUNNING";
  els.backendDetail.textContent = kind === "image" ? "图片上传检测中" : "视频上传检测中";

  const form = new FormData();
  form.append("file", file);
  appendModelId(form);

  try {
    if (kind === "video") {
      await streamUploadedVideo(form);
      return;
    }

    const data = await fetchJson(apiUrlWithModel("/api/detect/image"), {
      method: "POST",
      body: form,
      timeout: 15000,
    });

    applyResult(data, {
      appendEvent: true,
      keepVideoTimeline: kind === "video",
      storeMode: kind,
    });

    if (kind === "image" && state.imageAsset.result?.annotatedImage) {
      state.imageAsset.annotatedUrl = state.imageAsset.result.annotatedImage;
      els.fileImage.src = state.imageAsset.annotatedUrl;
    }

    if (kind === "video") {
      els.fileVideo.currentTime = 0;
      syncVideoResult(true);
      if (state.running) {
        els.fileVideo.play().catch(() => {});
      }
      els.backendDetail.textContent = "视频检测完成，结果随时间轴更新";
      setSourceHint(
        state.detections.length > 0
          ? `视频检测完成，当前帧命中 ${state.detections.length} 个目标`
          : "视频检测完成，拖动时间轴查看对应帧结果",
      );
    } else {
      els.backendDetail.textContent = "图片检测完成";
      setSourceHint(
        state.detections.length > 0
          ? `图片检测完成，命中 ${state.detections.length} 个目标`
          : "图片检测完成，当前未发现目标",
      );
    }
  } catch (error) {
    clearCurrentVisualResult();
    els.backendDetail.textContent = error.message;
    setSourceHint(`检测失败：${error.message}`);
  } finally {
    state.processing = false;
    els.liveChip.className = `live-chip ${state.connected ? "" : "offline"}`;
    els.liveChip.textContent = state.connected ? "RESULT" : "OFFLINE";
    checkHealth();
  }
}

async function streamUploadedVideo(form) {
  const controller = new AbortController();
  state.videoAsset.streamController = controller;
  state.videoAsset.streaming = true;
  updateVideoControlsVisibility();
  els.backendDetail.textContent = "视频上传中，等待后端开始流式检测";

  const response = await fetch(apiUrlWithModel("/api/detect/video/stream"), {
    method: "POST",
    body: form,
    signal: controller.signal,
  });

  if (!response.ok) {
    const text = await response.text();
    let message = text || `HTTP ${response.status}`;
    try {
      message = JSON.parse(text).error || message;
    } catch {}
    throw new Error(message);
  }

  if (!response.body) {
    throw new Error("browser does not support streaming response bodies");
  }

  let frames = 0;
  state.videoAsset.events = [];
  state.videoEventIndex = -1;
  els.backendDetail.textContent = "视频流式检测中";

  await readMjpegStream(response.body, async (jpegBytes, result) => {
    frames += 1;
    const frameUrl = URL.createObjectURL(new Blob([jpegBytes], { type: "image/jpeg" }));
    state.videoAsset.streamFrameUrl = frameUrl;
    state.videoAsset.currentAnnotatedImage = frameUrl;
    els.annotatedFrame.src = frameUrl;

    if (result) {
      result.annotated_image = frameUrl;
      const event = normalizeTimelineEvent(result, result.frame_width || 0, result.frame_height || 0);
      state.videoAsset.events.push(event);
      state.videoEventIndex = state.videoAsset.events.length - 1;
      applyVideoEvent(event);
    }

    els.backendDetail.textContent = `视频流式检测中 · ${frames} 帧`;
    setSourceHint(
      state.detections.length > 0
        ? `视频流式检测中，当前帧命中 ${state.detections.length} 个目标`
        : "视频流式检测中，当前帧未发现目标",
    );
  });

  state.videoAsset.streaming = false;
  state.videoAsset.streamController = null;
  updateVideoControlsVisibility();
  state.videoAsset.result = state.lastResult;
  els.fileVideo.currentTime = 0;
  syncVideoResult(true);
  if (state.running) {
    els.fileVideo.play().catch(() => {});
  }

  if (frames === 0) {
    throw new Error("视频没有可处理帧");
  }

  els.backendDetail.textContent = "视频流式检测完成";
  setSourceHint(
    state.detections.length > 0
      ? `视频流式检测完成，最后一帧命中 ${state.detections.length} 个目标`
      : "视频流式检测完成，未发现目标",
  );
}

function normalizePayload(payload) {
  const frameWidth = payload.frame_width || payload.width || payload.image_width || 0;
  const frameHeight = payload.frame_height || payload.height || payload.image_height || 0;
  const detections = (payload.detections || []).map((item) => normalizeDetection(item, frameWidth, frameHeight));
  const events = (payload.events || []).map((entry) => normalizeTimelineEvent(entry, frameWidth, frameHeight));

  return {
    detections,
    frameWidth,
    frameHeight,
    inferMs: payload.infer_ms ?? payload.infer_ms_avg ?? null,
    fps: payload.fps ?? null,
    time: payload.time || new Date().toLocaleTimeString("zh-CN", { hour12: false }),
    annotatedImage: payload.annotated_image || payload.annotatedImage || "",
    events,
  };
}

function normalizeTimelineEvent(entry, fallbackWidth, fallbackHeight) {
  const frameWidth = entry.frame_width || fallbackWidth || 0;
  const frameHeight = entry.frame_height || fallbackHeight || 0;
  return {
    frameIndex: entry.frame_index ?? 0,
    frameTime: Number(entry.frame_time ?? 0),
    frameWidth,
    frameHeight,
    inferMs: entry.infer_ms ?? null,
    fps: entry.fps ?? null,
    time: entry.time || "--",
    annotatedImage: entry.annotated_image || entry.annotatedImage || "",
    detections: (entry.detections || []).map((item) => normalizeDetection(item, frameWidth, frameHeight)),
  };
}

function normalizeDetection(item, frameWidth, frameHeight) {
  const label = item.label || item.class_name || item.name || String(item.class_id ?? "unknown");
  const meta = CLASS_META[label] || { zh: label, risk: "medium", color: "#b87912" };
  const score = Number(item.score ?? item.confidence ?? 0);
  let boxNorm = item.box_norm || item.normalized_box || null;

  if (!boxNorm && item.box && frameWidth > 0 && frameHeight > 0) {
    const [x1, y1, x2, y2] = item.box;
    boxNorm = [x1 / frameWidth, y1 / frameHeight, x2 / frameWidth, y2 / frameHeight];
  }

  return {
    label,
    zh: item.zh || meta.zh,
    risk: item.risk || meta.risk,
    color: item.color || meta.color,
    score,
    boxNorm,
  };
}

function applyResult(payload, options = {}) {
  const normalized = normalizePayload(payload);

  state.lastResult = {
    detections: normalized.detections,
    frameWidth: normalized.frameWidth,
    frameHeight: normalized.frameHeight,
    inferMs: normalized.inferMs,
    fps: normalized.fps,
    time: normalized.time,
    annotatedImage: normalized.annotatedImage,
  };
  state.detections = normalized.detections;
  els.resultTime.textContent = state.lastResult.time;

  if (options.keepVideoTimeline) {
    state.videoAsset.events = normalized.events;
    state.videoEventIndex = -1;
    syncVideoResult(true);
  } else {
    resetVideoTimeline();
  }

  if (options.storeMode === "image") {
    state.imageAsset.result = state.lastResult;
  }
  if (options.storeMode === "video") {
    state.videoAsset.result = state.lastResult;
  }
  if (options.storeMode === "stream") {
    state.streamResult = state.lastResult;
  }

  if (options.appendEvent !== false) pushEvent(state.lastResult);
}

function syncVideoResult(force = false) {
  if (state.mode !== "video" || !state.videoAsset.events.length) return;

  const currentTime = Number.isFinite(els.fileVideo.currentTime) ? els.fileVideo.currentTime : 0;
  let nextIndex = 0;
  for (let i = 0; i < state.videoAsset.events.length; i += 1) {
    if (state.videoAsset.events[i].frameTime <= currentTime + 0.05) nextIndex = i;
    else break;
  }

  if (!force && nextIndex === state.videoEventIndex) return;

  state.videoEventIndex = nextIndex;
  const event = state.videoAsset.events[nextIndex];
  applyVideoEvent(event);
}

function applyVideoEvent(event) {
  if (event.annotatedImage && state.videoAsset.currentAnnotatedImage !== event.annotatedImage) {
    state.videoAsset.currentAnnotatedImage = event.annotatedImage;
    els.annotatedFrame.src = event.annotatedImage;
  }
  state.detections = event.detections;
  state.lastResult = {
    detections: event.detections,
    frameWidth: event.frameWidth,
    frameHeight: event.frameHeight,
    inferMs: event.inferMs,
    fps: event.fps ?? state.videoAsset.result?.fps ?? null,
    time: event.time,
    annotatedImage: event.annotatedImage,
  };
  els.resultTime.textContent = state.lastResult.time;
  pushEvent(state.lastResult);
}

function draw() {
  const now = performance.now();
  const view = resizeCanvas();
  updateFps(now);

  if (state.mode === "video") {
    if (!state.videoAsset.streaming) {
      syncVideoResult();
    }
    updateVideoTimelineUi();
  }

  if (state.mode === "stream" && state.connected && state.running && els.streamImage.complete && els.streamImage.naturalWidth > 0) {
    drawMediaCover(els.streamImage, view);
  } else if (state.mode === "image" && els.fileImage.complete && els.fileImage.naturalWidth > 0) {
    drawMediaContain(els.fileImage, view);
  } else if (state.mode === "video" && state.lastResult?.annotatedImage && els.annotatedFrame.complete && els.annotatedFrame.naturalWidth > 0) {
    drawMediaCover(els.annotatedFrame, view);
  } else if (state.mode === "video" && els.fileVideo.readyState >= 2) {
    drawMediaCover(els.fileVideo, view);
  } else {
    drawPlaceholder(view);
  }

  if (!hasBackendAnnotatedVisual()) {
    drawDetections(state.detections, view);
  }
  renderPanel();
  requestAnimationFrame(draw);
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width: rect.width, height: rect.height };
}

function hasBackendAnnotatedVisual() {
  return (
    state.mode === "stream"
    || (state.mode === "image" && Boolean(state.imageAsset.result?.annotatedImage))
    || (state.mode === "video" && Boolean(state.lastResult?.annotatedImage))
  );
}

function updateCanvasAspect() {
  els.canvasWrap.style.setProperty("--media-aspect-ratio", "16 / 9");
}

function updateFps(now) {
  state.frameCount += 1;
  if (now - state.lastFpsTime >= 1000) {
    state.fps = state.frameCount;
    state.frameCount = 0;
    state.lastFpsTime = now;
  }
}

function updateVideoTimelineUi() {
  updateVideoControlsVisibility();
  if (state.mode !== "video" || !state.videoAsset.url) return;

  const duration = Number.isFinite(els.fileVideo.duration) ? els.fileVideo.duration : state.videoAsset.duration || 0;
  const currentTime = Number.isFinite(els.fileVideo.currentTime) ? els.fileVideo.currentTime : state.videoAsset.currentTime || 0;
  state.videoAsset.duration = duration;
  state.videoAsset.currentTime = currentTime;

  if (duration > 0 && !state.seekDragging) {
    els.videoSeek.value = String(Math.round((currentTime / duration) * 1000));
  }

  els.videoTime.textContent = `${formatTime(currentTime)} / ${formatTime(duration)}`;
}

function updateVideoControlsVisibility() {
  const visible = state.mode === "video" && Boolean(state.videoAsset.url) && !state.videoAsset.streaming;
  els.videoControls.classList.toggle("hidden", !visible);
}

function formatTime(value) {
  const total = Math.max(0, Math.floor(value || 0));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function drawMediaCover(media, view) {
  const w = view.width;
  const h = view.height;
  const sourceW = media.videoWidth || media.naturalWidth;
  const sourceH = media.videoHeight || media.naturalHeight;
  if (!sourceW || !sourceH) {
    drawPlaceholder(view);
    return;
  }

  const sourceRatio = sourceW / sourceH;
  const canvasRatio = w / h;
  let sx = 0;
  let sy = 0;
  let sw = sourceW;
  let sh = sourceH;

  if (sourceRatio > canvasRatio) {
    sw = sourceH * canvasRatio;
    sx = (sourceW - sw) / 2;
  } else {
    sh = sourceW / canvasRatio;
    sy = (sourceH - sh) / 2;
  }

  ctx.drawImage(media, sx, sy, sw, sh, 0, 0, w, h);
  ctx.fillStyle = "rgba(0, 0, 0, 0.12)";
  ctx.fillRect(0, 0, w, h);
}

function drawMediaContain(media, view) {
  const w = view.width;
  const h = view.height;
  const sourceW = media.videoWidth || media.naturalWidth;
  const sourceH = media.videoHeight || media.naturalHeight;
  if (!sourceW || !sourceH) {
    drawPlaceholder(view);
    return;
  }

  const scale = Math.min(w / sourceW, h / sourceH);
  const drawW = sourceW * scale;
  const drawH = sourceH * scale;
  const dx = (w - drawW) / 2;
  const dy = (h - drawH) / 2;

  ctx.fillStyle = "#111827";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(media, 0, 0, sourceW, sourceH, dx, dy, drawW, drawH);
}

function drawPlaceholder(view) {
  ctx.fillStyle = "#111827";
  ctx.fillRect(0, 0, view.width, view.height);
  ctx.fillStyle = "#dbe4ee";
  ctx.font = "700 24px Microsoft YaHei, Arial";
  const title = state.connected ? "等待真实视频帧" : "未连接开发板后端";
  const x = Math.max(18, Math.min(48, view.width * 0.06));
  ctx.fillText(title, x, view.height / 2 - 10);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "14px Microsoft YaHei, Arial";
  ctx.fillText(
    state.connected ? state.sourceHint : "请启动 Atlas 后端服务后重新连接",
    x,
    view.height / 2 + 26,
  );
}

function drawDetections(detections, view) {
  ctx.save();
  detections.forEach((det) => {
    if (!det.boxNorm) return;
    const [x1, y1, x2, y2] = det.boxNorm.map((value) => Math.min(1, Math.max(0, Number(value))));
    const left = x1 * view.width;
    const top = y1 * view.height;
    const width = Math.max(2, (x2 - x1) * view.width);
    const height = Math.max(2, (y2 - y1) * view.height);
    ctx.strokeStyle = det.color;
    ctx.lineWidth = 4;
    ctx.strokeRect(left, top, width, height);
    ctx.fillStyle = det.color;
    ctx.font = "700 16px Arial, sans-serif";
    const tag = `${det.label} ${(det.score * 100).toFixed(0)}%`;
    const tagWidth = ctx.measureText(tag).width + 24;
    ctx.fillRect(left, Math.max(0, top - 34), tagWidth, 30);
    ctx.fillStyle = "#fff";
    ctx.fillText(tag, left + 10, Math.max(22, top - 12));
  });
  ctx.restore();
}

function getRisk() {
  if (!state.connected && !state.lastResult) return { text: "未连接", cls: "idle" };
  if (state.detections.some((item) => item.risk === "high")) return { text: "高风险", cls: "high" };
  if (state.detections.length > 0) return { text: "中风险", cls: "medium" };
  return { text: "低风险", cls: "low" };
}

function renderPanel() {
  const risk = getRisk();
  els.riskBadge.className = `risk-badge ${risk.cls}`;
  els.riskBadge.textContent = risk.text;
  renderMetrics();

  if (!state.connected && !state.lastResult) return;
  renderAlerts("当前未发现告警目标");
}

function renderMetrics() {
  const maxConfidence = state.detections.reduce((max, item) => Math.max(max, item.score), 0);
  els.targetCount.textContent = state.connected || state.lastResult ? String(state.detections.length) : "--";
  els.confidence.textContent = state.connected || state.lastResult ? `${Math.round(maxConfidence * 100)}%` : "--";
  els.latency.textContent = state.lastResult?.inferMs != null ? `${Number(state.lastResult.inferMs).toFixed(1)} ms` : "--";
  els.fps.textContent =
    state.lastResult?.fps != null
      ? `${Number(state.lastResult.fps).toFixed(1)} FPS`
      : state.mode === "stream" && state.connected
        ? `${state.fps} FPS`
        : "--";
}

function pushEvent(result) {
  const risk = getRisk();
  const first = result.detections[0];
  const signature = `${result.time}-${first?.label || "none"}-${result.detections.length}`;
  if (state.eventHistory[0]?.signature === signature) return;

  state.eventHistory.unshift({
    signature,
    time: result.time,
    label: first?.zh ?? "无告警目标",
    confidence: first?.score ?? 0,
    risk: risk.text,
    count: result.detections.length,
  });
  state.eventHistory.splice(8);

  els.eventRows.innerHTML = state.eventHistory
    .map(
      (event) => `<div class="event-row"><span>${escapeHtml(event.time)}</span><strong>${escapeHtml(event.label)}</strong><span>${escapeHtml(event.risk)} · ${event.count} 个目标</span><div class="pill">${Math.round(event.confidence * 100)}%</div></div>`,
    )
    .join("");
}

function toggleRun() {
  if (state.processing) return;
  state.running = !state.running;
  els.toggleRun.textContent = state.running ? "||" : ">";
  els.toggleRun.title = state.running ? "暂停实时流" : "继续实时流";
  els.toggleRun.setAttribute("aria-label", els.toggleRun.title);

  if (state.mode === "stream") {
    if (state.running && state.connected) {
      stopAtlasStream();
      els.streamImage.src = apiUrlWithModel("/api/stream", { t: Date.now() });
      startEvents();
    } else {
      stopAtlasStream();
    }
  }

  if (state.mode === "video") {
    if (state.running) els.fileVideo.play().catch(() => {});
    else els.fileVideo.pause();
  }
}

async function saveApiBase() {
  state.apiBase = els.apiBase.value.trim() || DEFAULT_API;
  localStorage.setItem("atlas_api_base", state.apiBase);
  await loadModels().catch(() => {
    state.models = [];
    renderModelSelect();
  });
  checkHealth().then(() => {
    if (state.connected && state.mode === "stream") startAtlasStream();
  });
}

function handleModelChange() {
  state.modelId = els.modelSelect.value;
  localStorage.setItem("atlas_model_id", state.modelId);
  clearCurrentVisualResult();
  checkHealth().then(() => {
    if (state.mode === "stream" && state.running && state.connected && state.modelReady) {
      startAtlasStream();
    }
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[char]);
}

els.saveApi.addEventListener("click", saveApiBase);
els.modelSelect.addEventListener("change", handleModelChange);
els.refreshHealth.addEventListener("click", checkHealth);
els.useAtlasStream.addEventListener("click", startAtlasStream);
els.useImage.addEventListener("click", openImagePicker);
els.useVideo.addEventListener("click", openVideoPicker);
els.imageInput.addEventListener("change", (event) => uploadMedia(event.target.files[0], "image"));
els.videoInput.addEventListener("change", (event) => uploadMedia(event.target.files[0], "video"));
els.toggleRun.addEventListener("click", toggleRun);
els.fileImage.addEventListener("load", updateCanvasAspect);
els.streamImage.addEventListener("error", () => {
  if (state.mode === "stream") {
    clearCurrentVisualResult();
    setSourceHint("实时流不可用");
    els.backendDetail.textContent = "请确认 Atlas 摄像头和 /api/stream 已启动";
  }
});
els.fileVideo.addEventListener("timeupdate", () => {
  if (!state.videoAsset.streaming && !state.seekDragging) syncVideoResult();
  updateVideoTimelineUi();
});
els.fileVideo.addEventListener("loadedmetadata", () => updateVideoTimelineUi());
els.fileVideo.addEventListener("seeked", () => {
  if (!state.videoAsset.streaming) syncVideoResult(true);
  updateVideoTimelineUi();
});
els.fileVideo.addEventListener("ended", () => {
  if (!state.videoAsset.streaming) syncVideoResult(true);
  updateVideoTimelineUi();
});
els.videoSeek.addEventListener("pointerdown", () => {
  state.seekDragging = true;
});
els.videoSeek.addEventListener("input", () => {
  if (state.videoAsset.streaming) return;
  const duration = Number.isFinite(els.fileVideo.duration) ? els.fileVideo.duration : state.videoAsset.duration;
  if (!duration) return;
  const nextTime = (Number(els.videoSeek.value) / 1000) * duration;
  els.fileVideo.currentTime = nextTime;
  state.videoAsset.currentTime = nextTime;
  syncVideoResult(true);
  updateVideoTimelineUi();
});
els.videoSeek.addEventListener("change", () => {
  state.seekDragging = false;
  if (state.mode === "video" && state.running && !state.videoAsset.streaming) {
    els.fileVideo.play().catch(() => {});
  }
});

function updateClock() {
  els.clock.textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

setInterval(updateClock, 1000);
setInterval(() => {
  if (state.processing) return;
  if (state.mode === "stream" && state.running && state.connected) return;
  if (!state.connecting) checkHealth();
}, 5000);

setSource("stream", "atlas-camera://0", "Atlas 摄像头实时流", "等待 Atlas 后端提供真实检测结果");
setConnection("offline", "等待后端响应");
clearCurrentVisualResult();
updateVideoControlsVisibility();
updateClock();
loadModels()
  .catch(() => {
    state.models = [];
    renderModelSelect();
  })
  .then(() => checkHealth())
  .then(() => {
    if (state.connected && state.modelReady) startAtlasStream();
  });
requestAnimationFrame(draw);
