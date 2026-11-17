import { decodePacket, encodePacket, socketUrl } from "./transport.js";
import { createGuide, updateGuide, drawGuide } from "./calibration-ui.js";
import { createDepthView } from "./depth-view.js";
import { createRestLog } from "./rest-log.js";

/**
 * @template {HTMLElement} T
 * @param {string} id
 * @param {new() => T} kind
 * @returns {T}
 */
function element(id, kind) {
  const value = document.getElementById(id);

  if (!(value instanceof kind)) {
    throw new Error(`화면 요소를 확인해 주세요: ${id}`);
  }

  return value;
}

const join = element("join", HTMLElement);

const studio = element("studio", HTMLElement);

const phone = element("phone", HTMLElement);

const tokenInput = element("token", HTMLInputElement);

const roleInput = element("role", HTMLSelectElement);

const cameraChoice = element("view-camera", HTMLSelectElement);

const notice = element("notice", HTMLParagraphElement);

const localVideo = element("local-video", HTMLVideoElement);

const mainCanvas = element("main-canvas", HTMLCanvasElement);

const floorCanvas = element("floor-canvas", HTMLCanvasElement);

const statusText = element("connection", HTMLSpanElement);

const cameraIds = ["cam1", "cam2"];

const frames = new Map();

const thumbnails = new Map();

const guides = new Map();

const guideStates = new Map();

const phoneGuide = createGuide((action) => api(`/api/cameras/${roleInput.value}/${action}`));

element("phone-guide", HTMLDivElement).append(phoneGuide);

element("phone-restart", HTMLButtonElement).addEventListener("click", async () => {
  const button = element("phone-restart", HTMLButtonElement);
  button.disabled = true;

  try {
    const camera = await api(`/api/cameras/${roleInput.value}/guide-restart`);
    updateGuide(phoneGuide, camera.guide);
  } catch (error) {
    announce(error.message);
  } finally {
    button.disabled = false;
  }
});

const studioDepth = createDepthView(element("depth-canvas", HTMLCanvasElement), element("depth-frame-info", HTMLParagraphElement), element("depth-controls", HTMLDivElement));

const phoneDepth = createDepthView(element("phone-canvas", HTMLCanvasElement), element("phone-depth-info", HTMLParagraphElement), element("phone-depth-controls", HTMLDivElement));

let depthExpiryTimer;

let pairingCollapsed = false;

let socket;

let depthSocket;

let stream;

let pingTimer;

let renderTimer;

let captureTimer;

let active = false;

let generation = 0;

let wakeLock;

let key = sessionStorage.getItem("lava-key") || "";

const invitation = new URLSearchParams(location.hash.slice(1));

const invitedKey = invitation.get("key");

const invitedRole = invitation.get("role");

if (invitedKey) {
  key = invitedKey;
  sessionStorage.setItem("lava-key", key);
  element("key-field", HTMLDivElement).hidden = true;
}

if (cameraIds.includes(invitedRole) || invitedRole === "viewer") {
  roleInput.value = invitedRole;
}

if (location.hash) {
  history.replaceState(null, "", location.pathname + location.search);
}

tokenInput.value = key;

const restLog = createRestLog(element("rest-log", HTMLElement), () => key);

function announce(message) {
  notice.textContent = message;
  notice.hidden = !message;
}

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Session-Key": key },
    body: JSON.stringify(body || {}),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "요청을 처리하지 못했어요.");
  }

  return data;
}

function cameraButton(id, label, action) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.dataset.action = action;
  button.addEventListener("click", async () => {
    button.disabled = true;

    try {
      await api(`/api/cameras/${id}/${action}`);
      announce(`${id} · ${label} 요청을 적용했어요.`);
    } catch (error) {
      announce(error.message);
    } finally {
      button.disabled = false;
    }
  });

  return button;
}

for (const id of cameraIds) {
  const thumbnail = document.createElement("canvas");
  thumbnail.width = 320;
  thumbnail.height = 240;
  thumbnails.set(id, thumbnail);
  const thumb = document.createElement("button");
  thumb.type = "button";
  thumb.className = "thumb";
  const label = document.createElement("span");
  label.textContent = `카메라 ${id.slice(-1)}`;
  thumb.append(thumbnail, label);
  thumb.addEventListener("click", () => {
    cameraChoice.value = id;
    drawSelectedCamera();
  });
  document.querySelector("#thumbnails").append(thumb);
  const panel = document.createElement("article");
  panel.className = "camera-panel";
  panel.id = `panel-${id}`;
  const title = document.createElement("h3");
  title.textContent = `카메라 ${id.slice(-1)}`;
  const connection = document.createElement("strong");
  connection.className = "connection-state";
  connection.textContent = "연결 대기";
  const info = document.createElement("p");
  info.className = "camera-info";
  info.textContent = "연결 대기";
  const buttons = document.createElement("div");
  buttons.className = "buttons";

  for (const [labelText, action] of [
    ["보정 모드", "calibration"], ["1. 사진 추가", "sample"],
    ["2. 내부 보정", "intrinsics"], ["3. 바닥 위치 보정", "extrinsics"],
    ["4. 빈 배경 저장", "background"], ["5. 추적 시작", "play"],
    ["사진 초기화", "reset-samples"], ["이동·교체 후 재보정", "invalidate"],
  ]) {
    buttons.append(cameraButton(id, labelText, action));
  }

  const error = document.createElement("output");
  const guide = createGuide((action) => api(`/api/cameras/${id}/${action}`));
  guides.set(id, guide);
  const manual = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "수동 보정 · 다시 보정";
  buttons.append(cameraButton(id, "자동 보정 처음부터", "guide-restart"));
  manual.append(summary, buttons);
  panel.append(title, connection, guide, info, manual, error);
  document.querySelector("#camera-panels").append(panel);
}

function drawFloor(game, tracking) {
  const ctx = floorCanvas.getContext("2d");
  const cfg = game.config;
  const size = Math.min(288 / cfg.cols, 288 / cfg.rows);
  ctx.clearRect(0, 0, 320, 320);

  for (let row = 0; row < cfg.rows; row++) {
    for (let col = 0; col < cfg.cols; col++) {
      const hazard = game.cells.includes(row * cfg.cols + col);
      ctx.fillStyle = hazard ? (game.phase === "lava" ? "#d84b3e" : "#d89731") : "#344733";
      ctx.fillRect(16 + col * size, 16 + row * size, size - 2, size - 2);

      if (hazard) {
        ctx.fillStyle = "#ffffff";
        ctx.font = "16px system-ui";
        ctx.fillText(game.phase === "lava" ? "용암" : "경고", 20 + col * size, 40 + row * size);
      }
    }
  }

  ctx.fillStyle = tracking.valid ? "#ffffff" : "#a6b0bf";

  for (const foot of tracking.feet) {
    const x = 16 + (foot[0] - cfg.origin_x) / cfg.cell_m * size;
    const y = 16 + (foot[1] - cfg.origin_y) / cfg.cell_m * size;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
  }
}

function updateState(next) {
  if (!pairingCollapsed && next.cameras.slice(0, 2).every((camera) => camera.connection_status === "receiving")) {
    element("pairing", HTMLDetailsElement).open = false;
    pairingCollapsed = true;
  }

  document.querySelector("#depth-status").textContent = `${next.depth.backend} · ${next.depth.error || `${next.depth.resolution.join("×")} · 처리 ${next.depth.processing_ms}ms`}`;
  document.querySelector("#time").textContent = `${next.game.remaining_s.toFixed(1)}초`;
  document.querySelector("#lives").textContent = String(next.game.lives);
  const names = { ready: "게임 준비", playing: "플레이 중", paused: "추적 확인 중 · 시간 정지", won: "생존 성공", lost: "게임 오버" };
  document.querySelector("#game-status").textContent = names[next.game.status];
  document.querySelector("#tracking-status").textContent = next.game.reason || next.tracking.reason || `좌표 확인 중 · 시점 차이 ${Math.round(next.tracking.skew_ms || 0)}ms`;
  element("tracking-status", HTMLParagraphElement).hidden = next.tracking.reason === next.depth.error;
  element("start-game", HTMLButtonElement).disabled = !next.tracking.valid || ["playing", "paused"].includes(next.game.status);
  drawFloor(next.game, next.tracking);

  for (const camera of next.cameras) {
    const panel = document.querySelector(`#panel-${camera.id}`);
    const info = panel.querySelector(".camera-info");
    const labels = { disconnected: "연결 대기", waiting: "서버 연결됨 · 영상 확인 중", receiving: "영상 수신 중", stalled: "영상 끊김" };
    const connection = labels[camera.connection_status];
    const badge = panel.querySelector(".connection-state");
    badge.textContent = connection;
    badge.setAttribute("data-state", camera.connection_status);

    if (camera.connection_status !== "receiving") {
      frames.get(camera.id)?.close();
      frames.delete(camera.id);
      const thumbnail = thumbnails.get(camera.id);
      thumbnail.getContext("2d").clearRect(0, 0, thumbnail.width, thumbnail.height);
      const live = document.getElementById(`live-${camera.id}`);

      if (live instanceof HTMLCanvasElement) {
        live.getContext("2d").clearRect(0, 0, live.width, live.height);
        element(`live-status-${camera.id}`, HTMLParagraphElement).textContent = connection;
      }

      if (cameraChoice.value === camera.id) {
        mainCanvas.getContext("2d").clearRect(0, 0, mainCanvas.width, mainCanvas.height);
        element("empty-stage", HTMLParagraphElement).hidden = false;
      }
    }

    const calibration = camera.calibrated ? "바닥 위치 보정됨" : camera.intrinsics ? "내부 보정됨 · 바닥 위치 미확인" : "보정 필요";

    const liveStatus = document.getElementById(`live-status-${camera.id}`);

    if (liveStatus) {
      liveStatus.textContent = `${connection} · ${camera.fps} FPS · ${calibration}`;
    }

    info.textContent = `${camera.transport || "연결 전"} · ${camera.fps} FPS · 수신 ${camera.received_frames}장\n${calibration} · 유효 사진 ${camera.samples}/16장 이상\n처리 ${camera.processing_ms}ms · 왕복 ${camera.rtt_ms ?? "–"}ms · 영상 나이 ${camera.age_ms ?? "–"}ms`;
    panel.querySelector("output").textContent = camera.error;
    guideStates.set(camera.id, camera.guide);
    updateGuide(guides.get(camera.id), camera.guide);
  }

  const pairReady = next.cameras.slice(0, 2).every((camera) => camera.connection_status === "receiving" && camera.calibrated && camera.mode === "play" && camera.quality.passed);

  const depthChecklist = element("depth-readiness", HTMLUListElement);

  const readiness = [
    ["카메라 1 · 바닥 보정과 추적 시작", next.cameras[0].calibrated && next.cameras[0].mode === "play"],
    ["카메라 2 · 바닥 보정과 추적 시작", next.cameras[1].calibrated && next.cameras[1].mode === "play"],
    ["동시 기준판 · 좌우 정렬 검증", next.cameras.every((camera) => camera.quality.passed)],
    ["깊이 모델 준비", next.depth.ready],
    ["새 깊이 영상 수신", pairReady && Boolean(next.tracking.capture_ms)],
  ];

  depthChecklist.replaceChildren(...readiness.map(([label, done]) => {
    const item = document.createElement("li");
    item.textContent = `${done ? "✓" : "○"} ${label}`;
    item.dataset.done = String(done);

    return item;
  }));

  if (!pairReady) {
    studioDepth.clear(next.depth.error || "좌우 카메라를 연결하고 보정을 완료해 주세요.");
  }

  drawSelectedCamera();
}

function drawSelectedCamera() {
  const guide = guideStates.get(cameraChoice.value);

  if (guide) {
    const instruction = element("active-guide-status", HTMLParagraphElement);
    const message = `카메라 ${cameraChoice.value.slice(-1)} · ${guide.message}`;

    if (instruction.textContent !== message) {
      instruction.textContent = message;
    }
  }

  const frame = frames.get(cameraChoice.value);

  if (!frame) {
    element("empty-stage", HTMLParagraphElement).hidden = false;

    return;
  }

  mainCanvas.width = frame.width;
  mainCanvas.height = frame.height;
  mainCanvas.getContext("2d").drawImage(frame, 0, 0);
  drawGuide(mainCanvas.getContext("2d"), guideStates.get(cameraChoice.value), mainCanvas.width, mainCanvas.height);
  element("empty-stage", HTMLParagraphElement).hidden = true;
}

async function receiveCameraFrame(buffer) {
  const packet = decodePacket(buffer);
  const bitmap = await createImageBitmap(packet.blob.slice(0, packet.header.composite_jpeg_bytes));

  if (!active) {
    bitmap.close();

    return;
  }

  const id = packet.header.camera_id;
  frames.get(id)?.close();
  frames.set(id, bitmap);
  const thumbnail = thumbnails.get(id);
  thumbnail.getContext("2d").drawImage(bitmap, 0, 0, thumbnail.width, thumbnail.height);

  const live = document.getElementById(`live-${id}`);

  if (live instanceof HTMLCanvasElement) {
    const reference = await createImageBitmap(packet.blob.slice(packet.header.composite_jpeg_bytes));

    if (active) {
      live.width = reference.width;
      live.height = reference.height;
      live.getContext("2d").drawImage(reference, 0, 0);
    }

    reference.close();
  }

  drawSelectedCamera();
}

function startDepthStream(view) {
  const currentGeneration = generation;
  const connection = new WebSocket(socketUrl());
  depthSocket = connection;
  connection.binaryType = "arraybuffer";
  connection.onopen = () => connection.send(JSON.stringify({ role: "depth", token: key }));
  connection.onmessage = (event) => {
    if (active && generation === currentGeneration) {
      view.receive(event.data).catch((error) => announce(error.message));
    }
  };

  connection.onclose = () => {
    if (generation === currentGeneration) {
      view.clear("깊이 영상 연결이 끊겼어요. 다시 연결해 주세요.");
    }
  };

  depthExpiryTimer = setInterval(view.expire, 500);
}

async function startViewer() {
  const currentGeneration = generation;
  startDepthStream(studioDepth);
  socket = new WebSocket(socketUrl());
  socket.binaryType = "arraybuffer";
  let lastState = performance.now();
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("서버 응답이 없어요. 네트워크 연결을 확인해 주세요.")), 5000);
    socket.onopen = () => socket.send(JSON.stringify({ role: "viewer", token: key }));
    socket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        receiveCameraFrame(event.data).catch((error) => announce(error.message));

        return;
      }

      lastState = performance.now();
      updateState(JSON.parse(event.data));
      clearTimeout(timeout);
      resolve(undefined);
    };

    socket.onclose = () => {
      clearTimeout(timeout);
      reject(new Error("서버 연결이 끊겼어요."));

      if (active && generation === currentGeneration) {
        stop();
        statusText.textContent = "서버 연결 끊김";
        announce("서버 연결이 끊겼어요. ‘연결하기’를 눌러 다시 시도해 주세요.");
      }
    };
  });
  renderTimer = setInterval(() => {
    if (performance.now() - lastState > 3000) {
      stop();
      statusText.textContent = "서버 응답 없음";
      announce("서버 응답이 3초 동안 없어요. 네트워크를 확인하고 다시 연결해 주세요.");
    }
  }, 500);
  restLog.start();
}

async function startPhone(id) {
  if (!window.isSecureContext || !navigator.mediaDevices) {
    throw new Error("핸드폰 카메라는 신뢰된 HTTPS로 접속해야 해요.");
  }

  statusText.textContent = "카메라 권한 확인 중";

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 }, frameRate: { ideal: 30, max: 30 } },
    });
  } catch (error) {
    if (!["NotFoundError", "NotReadableError"].includes(error.name)) {
      throw error;
    }

    statusText.textContent = "카메라를 다시 확인 중";
    await new Promise((resolve) => setTimeout(resolve, 500));
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
  }

  localVideo.srcObject = stream;
  await localVideo.play();
  const settings = stream.getVideoTracks()[0].getSettings();
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  const ratio = localVideo.videoHeight / localVideo.videoWidth;
  canvas.height = ratio > 1 ? 640 : ratio > 0.65 ? 480 : 360;
  const ctx = canvas.getContext("2d", { alpha: false });
  const profile = [stream.getVideoTracks()[0].label, settings.deviceId, settings.width, settings.height, `640x${canvas.height}`].join("|").slice(0, 240);
  const initialWidth = localVideo.videoWidth;
  const initialHeight = localVideo.videoHeight;
  const currentGeneration = generation;
  let seq = 0;
  let inFlight = false;
  let synced = false;
  let lastResult = performance.now();
  let sendStarted = 0;
  socket = new WebSocket(socketUrl());
  socket.binaryType = "arraybuffer";

  const ping = () => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "ping", client_ms: performance.now() }));
    }
  };

  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("서버의 연결 확인 응답이 없어요. 네트워크를 확인해 주세요.")), 5000);
    socket.onopen = () => socket.send(JSON.stringify({ role: "camera", camera_id: id, token: key }));
    socket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        const packet = decodePacket(event.data);
        updateGuide(phoneGuide, packet.header.guide);
        const overlay = element("phone-scan-overlay", HTMLCanvasElement);
        overlay.width = localVideo.videoWidth;
        overlay.height = localVideo.videoHeight;
        drawGuide(overlay.getContext("2d"), packet.header.guide, overlay.width, overlay.height);
        inFlight = false;
        lastResult = performance.now();
        statusText.textContent = "영상 수신 확인됨";
        document.querySelector("#phone-status").textContent = "영상 수신 중 · 렌즈 보정 때는 움직이고, 바닥 보정 뒤에는 고정해 주세요.";
        document.querySelector("#phone-metrics").textContent = `서버 처리 ${Math.round(packet.header.processing_ms)}ms · 응답 ${Math.round(lastResult - sendStarted)}ms · 수신 확인 ${packet.header.seq + 1}장`;

        return;
      }

      const message = JSON.parse(event.data);

      if (message.type === "ready") {
        clearTimeout(timeout);
        statusText.textContent = "서버 연결됨 · 영상 확인 중";
        ping();
        resolve(undefined);
      } else if (message.type === "pong") {
        socket.send(JSON.stringify({ type: "sync", client_send: message.client_ms, client_receive: performance.now() }));
        synced = true;
      } else if (message.type === "ack") {
        inFlight = false;
      } else if (message.type === "error") {
        inFlight = false;
        statusText.textContent = "영상 처리 오류";
        document.querySelector("#phone-status").textContent = message.message;
      }
    };

    socket.onclose = (event) => {
      clearTimeout(timeout);
      const reason = event.reason || "영상 연결이 끊겼어요. 다시 연결해 주세요.";
      reject(new Error(reason));

      if (active && generation === currentGeneration) {
        stop();
        statusText.textContent = "연결 끊김";
        announce(reason);
      }
    };
  });
  pingTimer = setInterval(ping, 5000);
  captureTimer = setInterval(async () => {
    if (!active || generation !== currentGeneration) {
      return;
    }

    if (performance.now() - lastResult > 3000) {
      stop();
      statusText.textContent = "영상 응답 없음";
      announce("영상 응답이 3초 동안 없어요. 네트워크를 확인하고 다시 연결해 주세요.");

      return;
    }

    if (!synced || inFlight || socket.readyState !== WebSocket.OPEN || socket.bufferedAmount > 100000) {
      return;
    }

    if (localVideo.videoWidth !== initialWidth || localVideo.videoHeight !== initialHeight) {
      stop();
      announce("촬영 방향·해상도가 바뀌었어요. 다시 연결하고 보정해 주세요.");

      return;
    }

    inFlight = true;
    const captured = performance.now();
    ctx.drawImage(localVideo, 0, 0, canvas.width, canvas.height);

    try {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));

      if (!(blob instanceof Blob)) {
        throw new Error("카메라 영상을 만들지 못했어요.");
      }

      const packet = await encodePacket({ seq: seq++, captured_ms: captured, profile }, blob);

      if (!active || generation !== currentGeneration || socket.readyState !== WebSocket.OPEN) {
        return;
      }

      sendStarted = performance.now();
      socket.send(packet);
    } catch (error) {
      inFlight = false;
      announce(error.message);
    }
  }, 66);
  document.querySelector("#phone-title").textContent = `카메라 ${id.slice(-1)}`;
  const resultCanvas = element("phone-canvas", HTMLCanvasElement);
  resultCanvas.hidden = id !== "cam1";
  element("phone-depth", HTMLDivElement).hidden = id !== "cam1";

  if (id === "cam1") {
    startDepthStream(phoneDepth);
  }

  if ("wakeLock" in navigator) {
    try {
      wakeLock = await navigator.wakeLock.request("screen");
    } catch {
      announce("화면 자동 꺼짐을 해제하고 촬영해 주세요.");
    }
  }
}

function stop() {
  restLog.stop();
  pairingCollapsed = false;
  active = false;
  generation++;
  clearInterval(pingTimer);
  clearInterval(renderTimer);
  clearInterval(captureTimer);
  clearInterval(depthExpiryTimer);
  socket?.close();
  depthSocket?.close();
  studioDepth.clear();
  phoneDepth.clear();
  stream?.getTracks().forEach((track) => track.stop());
  wakeLock?.release();

  for (const frame of frames.values()) {
    frame.close();
  }

  frames.clear();

  localVideo.srcObject = null;
  localVideo.pause();
  join.hidden = false;
  studio.hidden = phone.hidden = true;
  statusText.textContent = "연결 종료";
}

document.querySelector("form#join-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = element("connect-button", HTMLButtonElement);
  button.disabled = true;
  key = tokenInput.value.trim();

  try {
    statusText.textContent = "서버 연결 확인 중";
    const response = await fetch("/api/state", { headers: { "X-Session-Key": key }, signal: AbortSignal.timeout(3000) });

    if (!response.ok) {
      element("key-field", HTMLDivElement).hidden = false;
      throw new Error("접속 키를 확인해 주세요.");
    }

    sessionStorage.setItem("lava-key", key);
    active = true;
    generation++;
    announce("");
    join.hidden = true;
    const role = roleInput.value;
    studio.hidden = role !== "viewer";
    phone.hidden = role === "viewer";

    if (role === "viewer") {
      await startViewer();
      loadPairing();
      statusText.textContent = "서버 연결됨";
    } else {
      await startPhone(role);
    }
  } catch (error) {
    stop();
    statusText.textContent = "연결 실패";

    const errors = {
      NotAllowedError: "카메라 사용을 허용한 뒤 다시 연결해 주세요.",
      NotFoundError: "사용 가능한 카메라를 찾지 못했어요. 카메라 연결을 확인하고 다시 시도해 주세요.",
      NotReadableError: "카메라를 시작하지 못했어요. 카메라를 사용하는 다른 앱을 닫고 다시 시도해 주세요.",
    };

    announce(errors[error.name] || error.message);
  } finally {
    button.disabled = false;
  }
});

for (const id of ["disconnect", "stop-camera"]) {
  document.getElementById(id).addEventListener("click", stop);
}

for (const action of ["start", "stop"]) {
  document.getElementById(`${action}-game`).addEventListener("click", async () => {
    try {
      updateState(await api(`/api/game/${action}`));
      announce("");
    } catch (error) {
      announce(error.message);
    }
  });
}

document.querySelector("form#config-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = element("config-form", HTMLFormElement);
  const config = Object.fromEntries([...new FormData(form)].map(([name, value]) => [name, Number(value)]));

  try {
    updateState(await api("/api/game/config", config));
    announce("게임 규칙을 저장했어요.");
  } catch (error) {
    announce(error.message);
  }
});

document.querySelector("button#fullscreen").addEventListener("click", () => {
  document.querySelector("#stage-container").requestFullscreen().catch((error) => announce(error.message));
});

window.addEventListener("pagehide", stop);

document.querySelector("button#download-depth").addEventListener("click", async () => {
  try {
    const response = await fetch("/api/depth/frame.npz", { headers: { "X-Session-Key": key } });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail);
    }

    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "depth-frame.npz";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    announce(error.message);
  }
});


cameraChoice.addEventListener("change", drawSelectedCamera);

const pairingUrls = [];

let pairingGeneration = 0;

async function showPairingCodes() {
  const version = ++pairingGeneration;
  const address = element("pair-address", HTMLSelectElement).value;

  const blobs = await Promise.all(cameraIds.map(async (id) => {
    const response = await fetch(`/api/pairing/${id}.png?address=${encodeURIComponent(address)}`, { headers: { "X-Session-Key": key } });

    if (!response.ok) {
      throw new Error("QR을 만들지 못했어요. 노트북 주소를 다시 확인해 주세요.");
    }

    return response.blob();
  }));

  if (version !== pairingGeneration) {
    return;
  }

  for (const url of pairingUrls) {
    URL.revokeObjectURL(url);
  }

  pairingUrls.length = 0;
  const container = element("pairing-codes", HTMLDivElement);
  container.replaceChildren();

  for (let index = 0; index < cameraIds.length; index++) {
    const id = cameraIds[index];
    const url = URL.createObjectURL(blobs[index]);
    pairingUrls.push(url);
    const card = document.createElement("article");
    const title = document.createElement("h3");
    title.textContent = `카메라 ${id.slice(-1)}${id === "cam1" ? " · 왼쪽" : " · 오른쪽"}`;
    const image = document.createElement("img");
    image.alt = `${title.textContent} 접속 QR`;
    image.src = url;
    image.width = image.height = 200;
    const link = document.createElement("a");
    const destination = new URL(location.origin);
    destination.hostname = address;
    destination.hash = new URLSearchParams({ key, role: id }).toString();
    link.href = destination.href;
    link.textContent = "접속 링크 열기";
    card.append(title, image, link);
    container.append(card);
  }
}

async function loadPairing() {
  const button = element("pair-cameras", HTMLButtonElement);
  const status = element("pairing-status", HTMLParagraphElement);
  button.disabled = true;
  status.textContent = "QR을 준비하고 있어요.";

  try {
    const response = await fetch("/api/pairing", { headers: { "X-Session-Key": key } });

    if (!response.ok) {
      throw new Error("서버에 다시 연결해 주세요.");
    }

    const pairing = await response.json();
    const addresses = element("pair-address", HTMLSelectElement);
    addresses.replaceChildren();

    for (const address of pairing.addresses) {
      const option = document.createElement("option");
      option.value = address;
      option.textContent = address === "127.0.0.1" ? `${address} · 노트북 전용` : address;
      addresses.append(option);
    }

    await showPairingCodes();
    status.textContent = "왼쪽 핸드폰은 카메라 1, 오른쪽 핸드폰은 카메라 2 QR을 스캔해 주세요.";
  } catch (error) {
    status.textContent = `${error.message} ‘QR 새로고침’을 눌러 다시 시도해 주세요.`;
  } finally {
    button.disabled = false;
  }
}

element("pair-cameras", HTMLButtonElement).addEventListener("click", loadPairing);

element("pair-address", HTMLSelectElement).addEventListener("change", () => {
  showPairingCodes().catch((error) => announce(error.message));
});

element("check-connection", HTMLButtonElement).addEventListener("click", async () => {
  const output = element("connection-check", HTMLParagraphElement);
  const button = element("check-connection", HTMLButtonElement);
  button.disabled = true;
  output.textContent = "서버 연결을 확인하고 있어요.";

  try {
    const started = performance.now();
    const response = await fetch("/api/health", { signal: AbortSignal.timeout(3000) });

    if (!response.ok) {
      throw new Error("서버 응답 오류");
    }

    output.textContent = `서버 접속 가능 · 응답 ${Math.round(performance.now() - started)}ms. ‘연결하기’를 누르면 영상 수신까지 확인해요.`;
  } catch {
    output.textContent = "서버에 접속하지 못했어요. 서버 실행·인증서·네트워크를 확인해 주세요. 학교망의 기기 간 차단은 SSL로 해제할 수 없어요.";
  } finally {
    button.disabled = false;
  }
});

if (invitedKey && invitedRole === "viewer") {
  element("join-form", HTMLFormElement).requestSubmit();
}
