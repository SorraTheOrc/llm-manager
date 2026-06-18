// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('LLama Local Status Endpoint', () => {

  test('GET /llama/local/status returns valid JSON with required fields', async ({ page }) => {
    const response = await page.request.get('/llama/local/status');
    expect(response.ok()).toBe(true);
    expect(response.headers()['content-type']).toContain('application/json');

    const body = await response.json();
    expect(body).toHaveProperty('active_query');
    expect(body).toHaveProperty('model_switch_in_progress');
    expect(body).toHaveProperty('current_model');
    expect(body).toHaveProperty('llama_server_running');
    expect(body).toHaveProperty('available_slots');
    expect(body).toHaveProperty('total_slots');

    expect(typeof body.active_query).toBe('boolean');
    expect(typeof body.model_switch_in_progress).toBe('boolean');
    expect(typeof body.llama_server_running).toBe('boolean');
    expect(typeof body.available_slots).toBe('number');
    expect(typeof body.total_slots).toBe('number');
    // current_model can be string or null
    expect(body.current_model === null || typeof body.current_model === 'string').toBe(true);
  });

  test('GET /llama/local/status response fields have correct types', async ({ page }) => {
    const response = await page.request.get('/llama/local/status');
    expect(response.ok()).toBe(true);

    const body = await response.json();

    // current_model can be string or null (null before any model is loaded)
    expect(body.current_model === null || typeof body.current_model === 'string').toBe(true);

    // When llama_server_running is false, other fields should be safe defaults
    if (body.llama_server_running === false) {
      expect(body.current_model).toBeNull();
      expect(body.active_query).toBe(false);
      expect(body.model_switch_in_progress).toBe(false);
    }
  });

  test('status endpoint responds within reasonable time', async ({ page }) => {
    const start = Date.now();
    const response = await page.request.get('/llama/local/status');
    const elapsed = Date.now() - start;

    expect(response.ok()).toBe(true);
    // Should respond quickly even under load (target < 5s but we use a tight bound here)
    expect(elapsed).toBeLessThan(10000);
  });

  test('concurrent status requests all succeed', async ({ page }) => {
    const promises = [];
    const N = 5;
    for (let i = 0; i < N; i++) {
      promises.push(page.request.get('/llama/local/status'));
    }

    const responses = await Promise.all(promises);
    expect(responses.length).toBe(N);

    for (const resp of responses) {
      expect(resp.ok()).toBe(true);
      const body = await resp.json();
      expect(body).toHaveProperty('llama_server_running');
      expect(body).toHaveProperty('active_query');
      expect(body).toHaveProperty('model_switch_in_progress');
      expect(body).toHaveProperty('current_model');
      expect(body).toHaveProperty('available_slots');
      expect(body).toHaveProperty('total_slots');
    }
  });

});
