export function createDepthView(canvas, info, controls) {
  controls.className = "depth-controls";
  controls.innerHTML = `
    <label>깊이 보기<select aria-label="깊이 표시 방식"><option value="overlay">현실 영상 위에 깊이 색상</option><option value="color">깊이 색상만</option><option value="reference">원본 영상</option></select></label>
    <div class="depth-legend" aria-label="깊이 색상 범례: 빨강 0.1m, 노랑 2m, 초록 4m, 파랑 8m 이상">
      <div class="depth-scale"></div><div class="depth-ticks"><span>0.1m · 가까움</span><span>2m</span><span>4m</span><span>8m 이상</span></div>
    </div><p class="muted">측정된 곳에만 색을 입혀요. 측정되지 않은 곳은 원본을 유지하고, ‘깊이 색상만’에서는 검게 보여요. 거리는 카메라 1의 앞쪽 깊이이며, 8~15m는 같은 파란색이에요.</p>`;
  const select = controls.querySelector("select");
  const overlay = document.createElement("canvas");
  let preview;
  let reference;
  let lastReceived = 0;
  let version = 0;
  let receiving = false;

  function draw() {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!preview || !reference) {
      return;
    }

    ctx.drawImage(select.value === "color" ? preview : reference, 0, 0);

    if (select.value === "overlay") {
      ctx.globalAlpha = 0.55;
      ctx.drawImage(overlay, 0, 0);
      ctx.globalAlpha = 1;
    }

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = Math.max(1, canvas.width / 400);
    ctx.beginPath();
    ctx.moveTo(canvas.width / 2 - 8, canvas.height / 2);
    ctx.lineTo(canvas.width / 2 + 8, canvas.height / 2);
    ctx.moveTo(canvas.width / 2, canvas.height / 2 - 8);
    ctx.lineTo(canvas.width / 2, canvas.height / 2 + 8);
    ctx.stroke();
  }

  select.addEventListener("change", draw);

  function clear(message = "카메라 1·2를 보정하면 현실 영상 위에 깊이 색상을 보여줘요.") {
    version++;
    preview?.close();
    reference?.close();
    preview = reference = undefined;
    receiving = false;
    draw();
    info.textContent = message;
    info.dataset.state = "waiting";
  }

  async function receive(buffer) {
    const current = ++version;

    const size = new DataView(buffer).getUint32(0);
    const metadata = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, size)));
    const start = 4 + size + metadata.depth_png_bytes;
    const middle = start + metadata.reference_jpeg_bytes;
    const end = middle + metadata.preview_jpeg_bytes;

    const images = await Promise.all([
      createImageBitmap(new Blob([buffer.slice(start, middle)], { type: "image/jpeg" })),
      createImageBitmap(new Blob([buffer.slice(middle, end)], { type: "image/jpeg" })),
      createImageBitmap(new Blob([buffer.slice(end)], { type: "image/png" })),
    ]);

    if (current !== version) {
      images.forEach((bitmap) => bitmap.close());

      return;
    }

    reference?.close();
    preview?.close();
    [reference, preview] = images;
    canvas.width = preview.width;
    canvas.height = preview.height;
    overlay.width = preview.width;
    overlay.height = preview.height;
    const ctx = overlay.getContext("2d", { willReadFrequently: true });
    const mask = images[2];
    ctx.drawImage(mask, 0, 0);
    mask.close();
    const alpha = ctx.getImageData(0, 0, overlay.width, overlay.height);

    // Use the lossless validity mask so JPEG artifacts cannot color unmeasured pixels.
    for (let i = 0; i < alpha.data.length; i += 4) {
      alpha.data[i + 3] = alpha.data[i];
    }

    ctx.putImageData(alpha, 0, 0);
    ctx.globalCompositeOperation = "source-in";
    ctx.drawImage(preview, 0, 0);
    ctx.globalCompositeOperation = "source-over";
    lastReceived = performance.now();
    receiving = true;
    const center = metadata.center_depth_m == null ? "측정 불가" : `${metadata.center_depth_m.toFixed(2)}m`;
    const coverage = metadata.valid_percent ?? "–";
    const warning = metadata.valid_percent < 10 ? "깊이 측정 부족 · 두 카메라가 같은 장면을 보도록 나란히 고정하고, 같은 기준판으로 바닥 위치를 다시 보정해 주세요. " : "";
    info.textContent = `${warning}중앙 십자선 ${center} · 색상으로 측정된 영역 ${coverage}% · 처리 ${Math.round(metadata.processing_ms)}ms · 카메라 1 기준`;
    info.dataset.state = warning ? "warning" : "receiving";
    draw();
  }

  function expire() {
    if (receiving && performance.now() - lastReceived > 1500) {
      clear("새 깊이 영상이 없어요. 카메라 연결·고정 상태를 확인해 주세요.");
    }
  }

  return { receive, clear, expire };
}
