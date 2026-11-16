export function createGuide(onAction) {
  const guide = document.createElement("div");
  guide.className = "calibration-guide";
  guide.innerHTML = `
    <div class="scan-summary">
      <canvas class="scan-ring" width="160" height="160" aria-label="보정 진행률"></canvas>
      <div><strong class="scan-stage">자동 보정 안내</strong><p class="scan-instruction" role="status" aria-live="polite">카메라를 연결해 주세요.</p></div>
    </div>
    <div class="scan-coverage" aria-label="자동 촬영 요구사항"></div>
    <ul class="scan-checklist" aria-label="카메라 보정 체크리스트"></ul>
    <button class="scan-action" type="button" hidden></button>`;
  const action = guide.querySelector("button");
  const errorText = document.createElement("p");
  errorText.className = "scan-error";
  errorText.setAttribute("role", "alert");
  errorText.hidden = true;
  guide.append(errorText);
  action.addEventListener("click", async () => {
    action.disabled = true;
    guide.dataset.busy = "true";
    errorText.hidden = true;

    try {
      const camera = await onAction(action.dataset.action);
      updateGuide(guide, camera.guide);
    } catch (error) {
      errorText.textContent = error.message;
      errorText.hidden = false;
    } finally {
      guide.dataset.busy = "false";
      action.disabled = guide.dataset.canAct !== "true";
    }
  });

  return guide;
}

export function updateGuide(container, guide) {
  if (!guide) {
    return;
  }

  const names = { collect: "움직이며 렌즈 보정", place: "이제 카메라를 고정해 주세요", floor: "바닥 위치 확인 중", background: "빈 배경 확인", ready: "보정 완료", manual: "수동 보정" };
  container.dataset.stage = guide.stage;
  container.querySelector(".scan-stage").textContent = names[guide.stage];
  const instruction = container.querySelector(".scan-instruction");

  if (instruction.textContent !== guide.message) {
    instruction.textContent = guide.message;
  }

  const canvas = container.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, 160, 160);

  for (let index = 0; index < 24; index++) {
    ctx.beginPath();
    ctx.arc(80, 80, 64, index / 24 * Math.PI * 2 - Math.PI / 2, (index + 0.7) / 24 * Math.PI * 2 - Math.PI / 2);
    ctx.strokeStyle = index < Math.round(guide.progress * 24) ? "#b9e97b" : "#364252";
    ctx.lineWidth = 10;
    ctx.lineCap = "round";
    ctx.stroke();
  }

  ctx.fillStyle = "#edf0f4";
  ctx.font = "600 28px system-ui";
  ctx.textAlign = "center";
  ctx.fillText(guide.stage === "ready" ? "✓" : `${Math.round(guide.progress * 100)}%`, 80, 90);
  canvas.setAttribute("aria-label", `보정 진행률 ${Math.round(guide.progress * 100)}%`);
  const list = container.querySelector(".scan-checklist");

  if (!list.children.length) {
    for (const check of guide.checks) {
      const item = document.createElement("li");
      item.dataset.check = check.id;
      list.append(item);
    }
  }

  for (const check of guide.checks) {
    const item = list.querySelector(`[data-check="${check.id}"]`);
    item.textContent = `${check.done ? "✓" : "○"} ${check.label}`;
    item.dataset.done = String(check.done);
  }

  const coverage = container.querySelector(".scan-coverage");
  coverage.hidden = guide.stage !== "collect";

  if (!coverage.children.length) {
    for (const check of guide.coverage) {
      const item = document.createElement("span");
      item.dataset.coverage = check.id;
      coverage.append(item);
    }
  }

  for (const check of guide.coverage) {
    const item = coverage.querySelector(`[data-coverage="${check.id}"]`);
    item.textContent = `${check.done ? "✓" : "○"} ${check.label}`;
    item.dataset.done = String(check.done);
  }

  const button = container.querySelector("button");
  button.hidden = !guide.action;
  button.dataset.action = guide.action || "";
  button.textContent = guide.action_label || "";
  container.dataset.canAct = String(guide.can_act);
  button.disabled = !guide.can_act || container.dataset.busy === "true";
}

export function drawGuide(ctx, guide, width, height) {
  if (!guide?.enabled || !["collect", "floor"].includes(guide.stage)) {
    return;
  }

  ctx.save();
  ctx.lineWidth = Math.max(2, width / 300);
  ctx.strokeStyle = "#ffffff80";
  ctx.setLineDash([8, 8]);
  ctx.strokeRect(width * 0.15, height * 0.12, width * 0.7, height * 0.76);
  ctx.setLineDash([]);

  if (guide.outline.length === 4) {
    ctx.strokeStyle = guide.stable ? "#b9e97b" : "#ffca5c";
    ctx.beginPath();
    guide.outline.forEach(([x, y], index) => {
      if (index === 0) {
        ctx.moveTo(x * width, y * height);
      } else {
        ctx.lineTo(x * width, y * height);
      }
    });
    ctx.closePath();
    ctx.stroke();
  }

  ctx.restore();
}
