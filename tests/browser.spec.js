import { test, expect, chromium } from "@playwright/test";

const phoneBrowsers = [];

async function phoneContext() {
  // Each phone has its own browser and camera service in actual use.
  const browser = await chromium.launch({
    channel: "chrome",
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
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

test("three phone browsers send video and receive independent live calibration checklists", async ({ browser }) => {
  const errors = [];
  const monitor = await browser.newContext();
  const page = await monitor.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.getByLabel("접속 키").fill("browser-key");
  await page.getByRole("button", { name: "연결하기", exact: true }).click();
  await expect(page.locator("#connection")).toHaveText("서버 연결됨", { timeout: 15000 });
  const contexts = [];

  for (const id of ["cam1", "cam2", "cam3"]) {
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

    if (id !== "cam3") {
      await expect(page.locator(`#live-status-${id}`)).toContainText("영상 수신 중");
      expect(await page.locator(`#live-${id}`).evaluate((canvas) => canvas.width)).toBe(640);
    }

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


test("desktop shows three QR codes above the game without a button click", async ({ page }) => {
  await page.goto("/#key=browser-key&role=viewer");
  await expect(page.locator("#pairing-codes img")).toHaveCount(3);
  await expect(page.locator("#pairing-codes img").first()).toBeInViewport();
  expect(await page.locator("#pairing-codes img").evaluateAll((images) => images.every((image) => image instanceof HTMLImageElement && image.complete && image.naturalWidth > 0))).toBe(true);
});

test("QR invitation supplies the key and one click starts phone video", async ({ browser }) => {
  const laptopContext = await browser.newContext();
  const laptop = await laptopContext.newPage();
  await laptop.goto("/#key=browser-key&role=viewer");
  await expect(laptop.locator("#connection")).toHaveText("서버 연결됨");
  await expect(laptop).toHaveURL("http://127.0.0.1:8000/");
  await expect(laptop.locator("#pairing-codes img")).toHaveCount(3);
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
      action: "guide-floor", action_label: "카메라 고정 완료 · 바닥 맞추기",
      checks: [
        { id: "video", label: "새 영상 수신", done: true },
        { id: "samples", label: "다양한 보정 사진 12/12장 이상", done: true },
        { id: "intrinsics", label: "렌즈 보정 · 오차 1.5px 이하", done: true },
        { id: "floor", label: "고정된 카메라의 바닥 위치 보정", done: false },
        { id: "background", label: "빈 배경 저장 · 추적 시작", done: false },
      ],
    };

    const guide = createGuide(async (action) => {
      if (action === "guide-floor") {
        state.stage = "background";
        state.message = "모든 카메라의 바닥 보정 후 사람이 화면에서 나가 주세요.";
        state.progress = 0.75;
        state.checks[3].done = true;
        state.action = "guide-background";
        state.action_label = "사람 없는 빈 배경 저장 · 깊이 시작";
      } else {
        state.stage = "ready";
        state.message = "보정 완료. 카메라를 고정한 채 유지해 주세요.";
        state.progress = 1;
        state.checks[4].done = true;
        state.action = null;
      }

      return { guide: state };
    });

    document.querySelector("#guide-fixture").append(guide);
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
    const view = createDepthView(document.querySelector("#metric-preview"), document.querySelector("#metric-info"), document.querySelector("#metric-controls"));
    const packet = await encodePacket({ depth_png_bytes: 0, reference_jpeg_bytes: reference.size, center_depth_m: 2.35, valid_percent: 85, processing_ms: 31 }, new Blob([reference, preview]));
    await view.receive(packet);
    document.querySelector("#metric-preview").addEventListener("clear-depth", () => view.clear("새 깊이 영상이 없어요."));
  });
  await expect(page.locator("#metric-info")).toContainText("2.35m");

  const pixels = await page.locator("#metric-preview").evaluate((canvas) => {
    const ctx = canvas.getContext("2d");

    return [Array.from(ctx.getImageData(20, 20, 1, 1).data), Array.from(ctx.getImageData(620, 20, 1, 1).data)];
  });

  expect(pixels[0][0]).toBeGreaterThan(pixels[0][2]);
  expect(pixels[1][2]).toBeGreaterThan(pixels[1][0]);
  await expect(page.locator("[data-check=floor]")).toHaveAttribute("data-done", "false");
  await page.getByRole("button", { name: "카메라 고정 완료 · 바닥 맞추기" }).click();
  await expect(page.locator("[data-check=floor]")).toHaveAttribute("data-done", "true");
  await expect(page.locator(".scan-instruction")).toContainText("사람이 화면에서 나가");
  await page.getByRole("button", { name: "사람 없는 빈 배경 저장 · 깊이 시작" }).click();
  await expect(page.locator(".scan-stage")).toHaveText("보정 완료");
  await page.getByLabel("깊이 표시 방식").selectOption("overlay");
  await expect(page.getByLabel("깊이 표시 방식")).toHaveValue("overlay");

  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.screenshot({ path: `data/depth-guide-${width}.png`, fullPage: true });
  }

  await page.getByLabel("깊이 표시 방식").selectOption("reference");
  await page.locator("#metric-preview").dispatchEvent("clear-depth");
  await expect(page.locator("#metric-info")).toContainText("새 깊이 영상이 없어요");
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
