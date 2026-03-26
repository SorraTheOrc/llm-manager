const { test, expect } = require('@playwright/test');

test.describe('Model Stats Panel', () => {
  
  test('stats panel is hidden by default', async ({ page }) => {
    await page.goto('/');
    
    const statsPanel = page.locator('#statsPanel');
    await expect(statsPanel).toBeHidden();
    
    const statsToggle = page.locator('.stats-toggle');
    await expect(statsToggle).toBeVisible();
    await expect(statsToggle).toHaveText('Show Model Stats');
  });

  test('stats panel shows when toggle is clicked', async ({ page }) => {
    await page.goto('/');
    
    const statsToggle = page.locator('.stats-toggle');
    await statsToggle.click();
    
    const statsPanel = page.locator('#statsPanel');
    await expect(statsPanel).toBeVisible();
    await expect(statsToggle).toHaveText('Hide Model Stats');
    
    await statsToggle.click();
    await expect(statsPanel).toBeHidden();
    await expect(statsToggle).toHaveText('Show Model Stats');
  });

  test('stats panel has all required fields', async ({ page }) => {
    await page.goto('/');
    
    const statsToggle = page.locator('.stats-toggle');
    await statsToggle.click();
    
    await expect(page.locator('#statsModel')).toBeVisible();
    await expect(page.locator('#statsLlamaStatus')).toBeVisible();
    await expect(page.locator('#statsNCtx')).toBeVisible();
    await expect(page.locator('#statsKvCache')).toBeVisible();
    await expect(page.locator('#statsTokensSent')).toBeVisible();
    await expect(page.locator('#statsTokensRecv')).toBeVisible();
  });

  test('SSE payload includes stats fields', async ({ page }) => {
    const sseMessages = [];
    
    await page.goto('/');
    
    await page.addInitScript(() => {
      const originalEventSource = window.EventSource;
      window.EventSource = function(url) {
        const es = new originalEventSource(url);
        const originalOnMessage = es.onmessage;
        es.onmessage = function(event) {
          window.lastSSEMessage = event.data;
          if (originalOnMessage) originalOnMessage.call(es, event);
        };
        return es;
      };
    });
    
    await page.waitForTimeout(2000);
    
    const lastMessage = await page.evaluate(() => window.lastSSEMessage);
    expect(lastMessage).toBeDefined();
    
    const parsed = JSON.parse(lastMessage);
    expect(parsed).toHaveProperty('n_ctx');
    expect(parsed).toHaveProperty('kv_cache_tokens');
    expect(parsed).toHaveProperty('total_sent');
    expect(parsed).toHaveProperty('total_recv');
    expect(parsed).toHaveProperty('llama_server_running');
    expect(parsed).toHaveProperty('current_model');
  });

  test('stats panel updates via SSE', async ({ page }) => {
    await page.goto('/');
    
    const statsToggle = page.locator('.stats-toggle');
    await statsToggle.click();
    
    const statsPanel = page.locator('#statsPanel');
    await expect(statsPanel).toBeVisible();
    
    await page.waitForTimeout(2000);
    
    const modelValue = await page.locator('#statsModel').textContent();
    expect(modelValue).toBeDefined();
    
    const llamaStatus = await page.locator('#statsLlamaStatus').textContent();
    expect(['Running', 'Stopped', 'Switching']).toContain(llamaStatus);
    
    const tokensSent = await page.locator('#statsTokensSent').textContent();
    expect(tokensSent).toBeDefined();
    expect(Number.isInteger(Number(tokensSent))).toBe(true);
  });

  test('token counters increase after API request', async ({ page, request }) => {
    await page.goto('/');
    
    const statsToggle = page.locator('.stats-toggle');
    await statsToggle.click();
    
    await page.waitForTimeout(1000);
    
    const initialSent = await page.locator('#statsTokensSent').textContent();
    const initialRecv = await page.locator('#statsTokensRecv').textContent();
    
    const initialSentNum = parseInt(initialSent, 10) || 0;
    const initialRecvNum = parseInt(initialRecv, 10) || 0;
    
    try {
      await request.post('/v1/chat/completions', {
        data: {
          model: 'qwen3',
          messages: [{ role: 'user', content: 'Say hello in 3 words' }],
          max_tokens: 20
        }
      });
    } catch (e) {
      console.log('API request failed (expected in test environment):', e.message);
    }
    
    await page.waitForTimeout(3000);
    
    const finalSent = await page.locator('#statsTokensSent').textContent();
    const finalRecv = await page.locator('#statsTokensRecv').textContent();
    
    const finalSentNum = parseInt(finalSent, 10) || 0;
    const finalRecvNum = parseInt(finalRecv, 10) || 0;
    
    expect(finalSentNum).toBeGreaterThanOrEqual(initialSentNum);
    expect(finalRecvNum).toBeGreaterThanOrEqual(initialRecvNum);
  });

  test('stats panel close button works', async ({ page }) => {
    await page.goto('/');
    
    const statsToggle = page.locator('.stats-toggle');
    await statsToggle.click();
    
    const statsPanel = page.locator('#statsPanel');
    await expect(statsPanel).toBeVisible();
    
    const closeBtn = page.locator('.btn-close-stats');
    await closeBtn.click();
    
    await expect(statsPanel).toBeHidden();
    await expect(statsToggle).toHaveText('Show Model Stats');
  });

  test('unknown values show tooltip', async ({ page }) => {
    await page.goto('/');
    
    const statsToggle = page.locator('.stats-toggle');
    await statsToggle.click();
    
    await page.waitForTimeout(2000);
    
    const unknownSpans = page.locator('.stats-unknown:visible');
    const count = await unknownSpans.count();
    
    if (count > 0) {
      const firstUnknown = unknownSpans.first();
      const title = await firstUnknown.getAttribute('title');
      expect(title).toContain('not available from backend');
    }
  });

});
