export function createRestLog(root, getKey) {
  root.innerHTML = `
    <div class="title-row"><div><p class="eyebrow">REST API · LIVE LOG</p><h2>실제로 반환되는 데이터</h2></div><div class="log-actions"><button type="button" data-log="toggle">로그 일시정지</button><button type="button" data-log="clear">로그 비우기</button></div></div>
    <p class="muted">GET /api/rgbd/frame · 응답 후 1초 간격으로 요청해요. 큰 Base64 데이터는 앞부분과 길이만 표시해요. 영상은 별도 실시간 연결로 계속 표시돼요.</p>
    <p class="rest-log-status" role="status" aria-live="polite">웹 연결 후 실제 API 응답을 확인해요.</p>
    <div class="rest-log-entries" aria-label="실제 REST 요청과 응답 기록"></div>`;
  const status = root.querySelector("p.rest-log-status");
  const entries = root.querySelector(".rest-log-entries");
  const toggle = root.querySelector("button[data-log=toggle]");
  let timer;
  let controller;
  let version = 0;
  let running = false;

  function record(code, elapsed, bytes, body) {
    const row = document.createElement("details");
    row.className = "rest-log-entry";
    row.dataset.status = String(code);
    const summary = document.createElement("summary");
    summary.textContent = `${new Date().toLocaleTimeString("ko-KR", { hour12: false })} · GET /api/rgbd/frame · ${code || "통신 실패"} · ${Math.round(elapsed)}ms · ${bytes.toLocaleString()}B`;
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(body, (name, value) => {
      if (name === "data_base64") {
        return `${value.slice(0, 48)}… [Base64 ${value.length.toLocaleString()}자, 화면에서 축약]`;
      }

      return value;
    }, 2);
    row.append(summary, pre);
    entries.prepend(row);

    while (entries.children.length > 20) {
      entries.lastElementChild.remove();
    }
  }

  function stop() {
    version++;
    running = false;
    clearTimeout(timer);
    controller?.abort();
    toggle.textContent = "로그 다시 시작";
    status.textContent = "로그 일시정지 · 기존 기록은 펼쳐서 확인할 수 있어요.";
  }

  function start() {
    stop();
    const current = version;
    running = true;
    toggle.textContent = "로그 일시정지";
    status.textContent = "실제 REST 응답을 요청하고 있어요.";

    async function poll() {
      const started = performance.now();
      const abort = new AbortController();
      controller = abort;
      const timeout = setTimeout(() => abort.abort(), 5000);

      try {
        const response = await fetch("/api/rgbd/frame", { headers: { "X-Session-Key": getKey() }, signal: abort.signal, cache: "no-store" });
        const raw = await response.text();

        if (current !== version) {
          return;
        }

        let body;

        try {
          body = JSON.parse(raw);
        } catch {
          body = { detail: raw.slice(0, 1000) || "빈 응답" };
        }

        record(response.status, performance.now() - started, new TextEncoder().encode(raw).byteLength, body);
        status.dataset.state = response.ok ? "success" : "error";
        status.textContent = response.ok
          ? `HTTP ${response.status} · 프레임 ${body.frame_id} · 깊이 ${body.depth?.shape?.join("×") || "–"} · 사람 마스크 ${body.segmentation?.available ? "포함" : "없음"}`
          : `HTTP ${response.status} · ${body.detail || "응답을 확인해 주세요."}`;
      } catch (error) {
        if (current === version) {
          record(0, performance.now() - started, 0, { detail: error.message });
          status.dataset.state = "error";
          status.textContent = "REST 응답을 받지 못했어요. 서버 연결과 아래 오류 기록을 확인해 주세요.";
        }
      } finally {
        clearTimeout(timeout);

        if (current === version && running) {
          timer = setTimeout(poll, 1000);
        }
      }
    }

    poll();
  }

  toggle.addEventListener("click", () => {
    if (running) {
      stop();
    } else {
      start();
    }
  });
  root.querySelector("button[data-log=clear]").addEventListener("click", () => entries.replaceChildren());

  return { start, stop };
}
