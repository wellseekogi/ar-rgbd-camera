import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testMatch: "browser.spec.js",
  workers: 1,
  timeout: 45000,
  use: {
    channel: "chrome",
    baseURL: "http://127.0.0.1:8000",
    launchOptions: { args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] },
    permissions: ["camera"],
  },
  webServer: {
    command: ".venv\\Scripts\\python.exe scripts/run.py --http-local",
    url: "http://127.0.0.1:8000/api/health",
    env: { LAVA_DISABLE_STEREO: "1", LAVA_SESSION_KEY: "browser-key" },
    reuseExistingServer: false,
    timeout: 30000,
  },
});
