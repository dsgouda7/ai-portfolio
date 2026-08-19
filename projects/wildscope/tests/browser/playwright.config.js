import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.WILDSCOPE_TEST_SERVER_URL || "http://127.0.0.1:5000";

export default defineConfig({
  testDir: ".",
  testMatch: "app.spec.js",
  timeout: 20_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } } },
    { name: "mobile", use: { ...devices["Pixel 7"] } },
  ],
});
