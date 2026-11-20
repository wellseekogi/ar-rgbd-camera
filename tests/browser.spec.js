import { test, expect, chromium } from "@playwright/test";
import { resolve } from "node:path";

const phoneBrowsers = [];

async function phoneContext() {
  // Each phone has its own browser and camera service in actual use.
  const browser = await chromium.launch({
    channel: "chrome",
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-video-capture=${resolve("data/browser-camera.y4m")}`],
  });

  phoneBrowsers.push(browser);

  return browser.newContext({ baseURL: "http://127.0.0.1:8000", permissions: ["camera"] });
}

test.afterEach(async () => {
  await Promise.all(phoneBrowsers.map((browser) => browser.close()));
  phoneBrowsers.length = 0;
});

test("desktop and mobile show the connection form without overflow", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "바닥을 게임판으로." })).toBeVisible();
  await page.getByLabel("접속 키").fill("wrong");
  await page.getByRole("button", { name: "연결하기", exact: true }).click();
  await expect(page.locator("#notice")).toContainText("접속 키");

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: `data/join-${width}.png`, fullPage: true });
  }
});

test("two phone browsers send video and receive independent live calibration checklists", async ({ browser }) => {
  const errors = [];
  const monitor = await browser.newContext();
  const page = await monitor.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.getByLabel("접속 키").fill("browser-key");
  await page.getByRole("button", { name: "연결하기", exact: true }).click();
  await expect(page.locator("#connection")).toHaveText("서버 연결됨", { timeout: 15000 });
  const contexts = [];

  for (const id of ["cam1", "cam2"]) {
    const context = await phoneContext();
    contexts.push(context);
    const phone = await context.newPage();
    phone.on("pageerror", (error) => errors.push(error.message));
    await phone.goto("/");
    await phone.getByLabel("접속 키").fill("browser-key");
    await phone.getByLabel("이 기기의 역할").selectOption(id);
    await phone.getByRole("button", { name: "연결하기", exact: true }).click();

    try {
      await expect(phone.locator("#connection")).toHaveText("영상 수신 확인됨", { timeout: 15000 });
      await expect(phone.locator("#phone-metrics")).toContainText("서버 처리", { timeout: 15000 });
    } catch (error) {
      console.log(await page.locator(`#panel-${id}`).innerText());
      console.log(await phone.locator("#phone-status").innerText());
      console.log(await phone.locator("#notice").innerText());
      console.log(await page.evaluate(async () => {
        const response = await fetch("/api/state", { headers: { "X-Session-Key": "browser-key" } });

        return response.json();
      }));
      throw error;
    }

    await expect(page.locator(`#panel-${id}`)).toContainText("WS");
    await expect(page.locator(`#panel-${id} .connection-state`)).toHaveText("영상 수신 중");
    await expect(page.locator(`#panel-${id} output`)).toBeEmpty();

    await expect(page.locator(`#live-status-${id}`)).toContainText("영상 수신 중");
    expect(await page.locator(`#live-${id}`).evaluate((canvas) => canvas.width)).toBe(640);

    await expect(phone.locator("#phone-guide [data-check=video]")).toHaveAttribute("data-done", "true");
    await expect(phone.locator("#phone-guide [data-check=intrinsics]")).toHaveAttribute("data-done", "false");
    await expect(phone.locator("#phone-guide .scan-instruction")).toContainText("기준판");
    await phone.setViewportSize({ width: 390, height: 1000 });
    expect(await phone.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await phone.screenshot({ path: `data/guide-phone-${id}.png`, fullPage: true });
  }

  await expect(page.getByRole("button", { name: "게임 시작하기" })).toBeDisabled();

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: `data/studio-${width}.png`, fullPage: true });
  }

  expect(errors).toEqual([]);

  for (const context of contexts) {
    await context.close();
  }

  await expect(page.locator("#panel-cam1")).toContainText("연결 대기", { timeout: 15000 });
  await monitor.close();
});


test("desktop shows two QR codes and only two camera controls", async ({ page }) => {
  await page.goto("/#key=browser-key&role=viewer");
  await expect(page.locator("#pairing-codes img")).toHaveCount(2);
  await expect(page.locator("#camera-panels .camera-panel")).toHaveCount(2);
  await expect(page.locator("#thumbnails .thumb")).toHaveCount(2);
  expect(await page.locator("#role option").evaluateAll((options) => options.map((option) => option.value))).toEqual(["viewer", "cam1", "cam2"]);
  expect(await page.locator("#view-camera option").evaluateAll((options) => options.map((option) => option.value))).toEqual(["cam1", "cam2"]);
  await expect(page.locator("#pairing-codes img").first()).toBeInViewport();
  expect(await page.locator("#pairing-codes img").evaluateAll((images) => images.every((image) => image instanceof HTMLImageElement && image.complete && image.naturalWidth > 0))).toBe(true);
});

test("QR invitation supplies the key and one click starts phone video", async ({ browser }) => {
  const laptopContext = await browser.newContext();
  const laptop = await laptopContext.newPage();
  await laptop.goto("/#key=browser-key&role=viewer");
  await expect(laptop.locator("#connection")).toHaveText("서버 연결됨");
  await expect(laptop).toHaveURL("http://127.0.0.1:8000/");
  await expect(laptop.locator("#pairing-codes img")).toHaveCount(2);
  await expect(laptop.locator("#pairing-codes img").first()).toBeInViewport();
  await laptop.locator("#pair-address").selectOption("127.0.0.1");
  const invitation = laptop.locator("#pairing-codes a").first();
  await expect(invitation).toHaveAttribute("href", /127\.0\.0\.1:8000.*role=cam1/);
  const link = await invitation.getAttribute("href");
  const cameraContext = await phoneContext();
  const phone = await cameraContext.newPage();
  await phone.goto(link);
  await expect(phone.locator("#key-field")).toBeHidden();
  await expect(phone.locator("#role")).toHaveValue("cam1");
  await expect(phone).toHaveURL("http://127.0.0.1:8000/");
  await phone.getByRole("button", { name: "서버 연결 확인하기" }).click();
  await expect(phone.locator("#connection-check")).toContainText("서버 접속 가능");
  await phone.getByRole("button", { name: "연결하기", exact: true }).click();
  await expect(phone.locator("#connection")).toHaveText("영상 수신 확인됨");
  await expect(laptop.locator("#panel-cam1 .connection-state")).toHaveText("영상 수신 중");

  for (const width of [1440, 390]) {
    await laptop.setViewportSize({ width, height: 1000 });
    expect(await laptop.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await laptop.screenshot({ path: `data/pairing-${width}.png`, fullPage: true });
  }

  await phone.getByRole("button", { name: "촬영 종료하기" }).click();
  await expect(laptop.locator("#panel-cam1 .connection-state")).toHaveText("연결 대기");
  await expect(laptop.locator("#empty-stage")).toBeVisible();
  await phone.getByRole("button", { name: "연결하기", exact: true }).click();
  await expect(phone.locator("#connection")).toHaveText("영상 수신 확인됨");
  await cameraContext.close();
  await laptopContext.close();
});

test("connection check reports an unreachable server", async ({ page }) => {
  await page.goto("/");
  await page.route("**/api/health", (route) => route.abort());
  await page.getByRole("button", { name: "서버 연결 확인하기" }).click();
  await expect(page.locator("#connection-check")).toContainText("서버에 접속하지 못했어요");
  await expect(page.locator("#connection")).not.toContainText("연결됨");
});

test("photo collection is visible without claiming calibration passed", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    const { createGuide, updateGuide } = await import("/static/calibration-ui.js");
    const guide = createGuide(async () => ({}));
    document.querySelector("main").replaceChildren(guide);

    const state = {
      enabled: true, stage: "collect", progress: 0, message: "잠시 멈추면 촬영해요.",
      checks: [{ id: "intrinsics", label: "렌즈 품질 검사", done: false }],
      coverage: [{ id: "center", label: "가운데", done: true }], quality: [],
      action: null, can_act: false, collection: { accepted: 4, minimum: 16, limit: 40, last_error: null },
    };

    updateGuide(guide, state);
    guide.addEventListener("quality-rejected", () => {
      state.collection.accepted = 18;
      state.collection.last_error = "별도 검증 오차가 커요.";
      updateGuide(guide, state);
    });
    guide.addEventListener("lens-verified", () => {
      state.stage = "place";
      state.progress = 0.25;
      updateGuide(guide, state);
    });
    guide.addEventListener("collection-limit", () => {
      state.collection.accepted = state.collection.limit;
      updateGuide(guide, state);
    });
  });
  await expect(page.getByLabel("유효 사진 4장 수집 · 최소 16장 · 렌즈 검증 미완료")).toBeVisible();
  await expect(page.locator(".scan-collection")).toContainText("유효 사진 4장 / 최소 16장");
  await expect(page.locator("[data-check=intrinsics]")).toHaveAttribute("data-done", "false");
  await page.locator(".calibration-guide").dispatchEvent("quality-rejected");
  await expect(page.locator(".scan-quality-error")).toContainText("별도 검증 오차가 커요");
  await expect(page.locator(".scan-collection")).toContainText("렌즈 품질 검사 통과가 필요");
  await expect(page.getByLabel("유효 사진 18장 수집 · 최소 16장 · 렌즈 검증 미완료")).toBeVisible();

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: `data/calibration-collection-${width}.png`, fullPage: true });
  }

  await page.locator(".calibration-guide").dispatchEvent("collection-limit");
  await expect(page.locator(".scan-quality-error")).toContainText("사진 초기화 · 다시 보정");
  await page.locator(".calibration-guide").dispatchEvent("lens-verified");
  await expect(page.getByLabel("검증 단계 완료율 25%")).toBeVisible();
  await expect(page.locator(".scan-quality-error")).toBeHidden();
});

test("guided checks advance and metric depth colors overlay the matching reference", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(async () => {
    const { createGuide, updateGuide } = await import("/static/calibration-ui.js");
    const { createDepthView } = await import("/static/depth-view.js");
    const { encodePacket } = await import("/static/transport.js");
    const main = document.querySelector("main");
    main.innerHTML = `<h1>카메라 보정과 실시간 깊이</h1><p class="muted">합성 데이터로 화면 동작 검증</p><div class="workspace"><section id="guide-fixture" class="camera-panel"></section><section><canvas id="metric-preview" width="640" height="480" aria-label="검증용 깊이맵"></canvas><p id="metric-info"></p><div id="metric-controls"></div></section></div>`;

    const state = {
      enabled: true, stage: "place", message: "같은 기준판을 바닥에 두고 카메라를 고정해 주세요.",
      progress: 0.5, coverage: [], outline: [], stable: true, can_act: true,
      quality: [{ label: "별도 영상 오차", value: "0.34px · 기준 0.8px", passed: true }],
      action: "guide-floor", action_label: "카메라 고정 완료 · 바닥 맞추기",
      checks: [
        { id: "video", label: "새 영상 수신", done: true },
        { id: "samples", label: "학습 12장 · 별도 검증 4장", done: true },
        { id: "intrinsics", label: "별도 영상 오차 0.8px 이하", done: true },
        { id: "floor", label: "고정된 카메라의 바닥 위치 보정", done: false },
        { id: "stereo", label: "좌우 정렬 · 기준판 크기 검증", done: false },
        { id: "background", label: "빈 배경 저장 · 추적 시작", done: false },
      ],
    };

    const guide = createGuide(async (action) => {
      if (action === "guide-floor") {
        state.stage = "verify";
        state.message = "좌우 동시 영상 검증을 기다려 주세요.";
        state.progress = 0.5;
        state.checks[3].done = true;
        state.action = null;
      } else {
        state.stage = "ready";
        state.message = "보정 완료. 카메라를 고정한 채 유지해 주세요.";
        state.progress = 1;
        state.checks[5].done = true;
        state.action = null;
      }

      return { guide: state };
    });

    document.querySelector("#guide-fixture").append(guide);
    guide.addEventListener("verified-pair", () => {
      state.stage = "background";
      state.message = "모든 카메라의 바닥 보정 후 사람이 화면에서 나가 주세요.";
      state.progress = 0.75;
      state.checks[4].done = true;
      state.action = "guide-background";
      state.action_label = "사람 없는 빈 배경 저장 · 깊이 시작";
      state.quality.push(
        { label: "바닥 고정 오차 RMS/p95", value: "0.30/0.58px · 기준 0.75/1.5px", passed: true },
        { label: "좌우 정렬 오차 p95", value: "0.42px · 기준 1px", passed: true },
        { label: "기준판 크기 오차 p95", value: "1.20% · 기준 5%", passed: true },
      );
      updateGuide(guide, state);
    });
    updateGuide(guide, state);
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#d0d0d0";
    ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = "#363636";
    ctx.fillRect(160, 100, 320, 280);
    const reference = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg"));
    const gradient = ctx.createLinearGradient(0, 0, 640, 0);
    gradient.addColorStop(0, "#ff5951");
    gradient.addColorStop(0.24, "#ffca5c");
    gradient.addColorStop(0.49, "#57d9b1");
    gradient.addColorStop(1, "#5d92ff");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 640, 480);
    const preview = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg"));
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 640, 480);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 400, 640, 80);
    const mask = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    const view = createDepthView(document.querySelector("#metric-preview"), document.querySelector("#metric-info"), document.querySelector("#metric-controls"));
    const metadata = { depth_png_bytes: 0, reference_jpeg_bytes: reference.size, preview_jpeg_bytes: preview.size, valid_mask_png_bytes: mask.size, center_depth_m: 2.35, valid_percent: 85, processing_ms: 31 };
    const packet = await encodePacket(metadata, new Blob([reference, preview, mask]));
    await view.receive(packet);
    document.querySelector("#metric-preview").addEventListener("clear-depth", () => view.clear("새 깊이 영상이 없어요."));
    document.querySelector("#metric-preview").addEventListener("sparse-depth", async () => {
      await view.receive(await encodePacket({ ...metadata, valid_percent: 0.4, center_depth_m: null }, new Blob([reference, preview, mask])));
    });
  });
  await expect(page.locator("#metric-info")).toContainText("2.35m");
  await expect(page.getByLabel("깊이 표시 방식")).toHaveValue("overlay");

  const pixels = await page.locator("#metric-preview").evaluate((canvas) => {
    const ctx = canvas.getContext("2d");

    return [Array.from(ctx.getImageData(20, 20, 1, 1).data), Array.from(ctx.getImageData(620, 20, 1, 1).data), Array.from(ctx.getImageData(20, 440, 1, 1).data)];
  });

  expect(pixels[0][0]).toBeGreaterThan(pixels[0][2]);
  expect(pixels[1][2]).toBeGreaterThan(pixels[1][0]);
  expect(pixels[2]).toEqual([208, 208, 208, 255]);
  await page.getByLabel("깊이 표시 방식").selectOption("reference");
  const original = await page.locator("#metric-preview").evaluate((canvas) => Array.from(canvas.getContext("2d").getImageData(20, 20, 1, 1).data));
  expect(original).toEqual([208, 208, 208, 255]);
  await page.getByLabel("깊이 표시 방식").selectOption("color");
  const color = await page.locator("#metric-preview").evaluate((canvas) => Array.from(canvas.getContext("2d").getImageData(20, 20, 1, 1).data));
  expect(color[0]).toBeGreaterThan(pixels[0][0]);
  await expect(page.locator("[data-check=floor]")).toHaveAttribute("data-done", "false");
  await page.getByRole("button", { name: "카메라 고정 완료 · 바닥 맞추기" }).click();
  await expect(page.locator("[data-check=floor]")).toHaveAttribute("data-done", "true");
  await expect(page.locator(".scan-stage")).toHaveText("좌우 정렬 검증");
  await expect(page.locator("[data-check=stereo]")).toHaveAttribute("data-done", "false");
  await expect(page.getByRole("button", { name: "사람 없는 빈 배경 저장 · 깊이 시작" })).toBeHidden();
  await page.locator(".calibration-guide").dispatchEvent("verified-pair");
  await expect(page.locator("[data-check=stereo]")).toHaveAttribute("data-done", "true");
  await expect(page.getByLabel("실측 품질 검사")).toContainText("0.42px");
  await expect(page.locator(".scan-instruction")).toContainText("사람이 화면에서 나가");
  await page.getByRole("button", { name: "사람 없는 빈 배경 저장 · 깊이 시작" }).click();
  await expect(page.locator(".scan-stage")).toHaveText("보정 검증 통과");
  await page.getByLabel("깊이 표시 방식").selectOption("overlay");
  await expect(page.getByLabel("깊이 표시 방식")).toHaveValue("overlay");

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: `data/depth-guide-${width}.png`, fullPage: true });
  }

  await page.getByLabel("깊이 표시 방식").selectOption("reference");
  await page.locator("#metric-preview").dispatchEvent("sparse-depth");
  await expect(page.locator("#metric-info")).toContainText("깊이 측정 부족");
  await expect(page.locator("#metric-info")).toContainText("0.4%");
  await page.locator("#metric-preview").dispatchEvent("clear-depth");
  await expect(page.locator("#metric-info")).toContainText("새 깊이 영상이 없어요");
  const cleared = await page.locator("#metric-preview").evaluate((canvas) => canvas.getContext("2d").getImageData(20, 20, 1, 1).data[3]);
  expect(cleared).toBe(0);
});

test("live monitor shows cameras depth and real REST success and failure logs", async ({ page }) => {
  let status = 200;
  let requests = 0;
  await page.route("**/api/rgbd/frame", async (route) => {
    requests++;
    await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(status === 200 ? {
      frame_id: "camera-1-test-frame", camera_id: "cam1",
      image: { encoding: "jpeg", data_base64: "A".repeat(300) },
      depth: { dtype: "float32", "shape": [480, 640], near_m: 0.1, far_m: 15, data_base64: "B".repeat(300) },
      segmentation: { available: true, model: "mediapipe_pose_landmarker_lite" },
    } : { detail: "새 깊이 결과가 없어요. 보정과 GPU 지연을 확인해 주세요." }) });
  });
  await page.goto("/#key=browser-key&role=viewer");
  await expect(page.locator(".rest-log-status")).toContainText("HTTP 200");
  await expect(page.locator("#live-cam1")).toBeVisible();
  await expect(page.locator("#live-cam2")).toBeVisible();
  await expect(page.locator("#depth-canvas")).toBeVisible();
  await page.locator(".rest-log-entry").first().locator("summary").click();
  await expect(page.locator(".rest-log-entry").first().locator("pre")).toContainText('"near_m": 0.1');
  await expect(page.locator(".rest-log-entry").first().locator("pre")).toContainText("Base64 300자");
  await expect(page.locator(".rest-log-entry").first().locator("pre")).not.toContainText("A".repeat(100));

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: `data/live-monitor-${width}.png`, fullPage: true });
  }

  status = 409;
  await expect(page.locator(".rest-log-status")).toContainText("HTTP 409", { timeout: 8000 });
  await page.getByRole("button", { name: "로그 일시정지", exact: true }).click();
  const pausedRequests = requests;
  await page.waitForTimeout(1300);
  expect(requests).toBe(pausedRequests);
  await page.getByRole("button", { name: "로그 비우기", exact: true }).click();
  await expect(page.locator(".rest-log-entry")).toHaveCount(0);
  await page.getByRole("button", { name: "로그 다시 시작", exact: true }).click();
  await expect(page.locator(".rest-log-entry")).toHaveCount(1);
});
