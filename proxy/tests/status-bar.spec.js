// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('LLama Proxy Status Bar', () => {
  
  test('page loads with status bar elements', async ({ page }) => {
    await page.goto('/');
    
    // Check status bar elements exist
    const currentModel = page.locator('#currentModelStatus');
    const llamaStatus = page.locator('#llamaServerStatus');
    const statusMessage = page.locator('#statusMessage');
    
    await expect(currentModel).toBeVisible();
    await expect(llamaStatus).toBeVisible();
    
    // Status message should be hidden initially
    await expect(statusMessage).toBeHidden();
    
    console.log('Current model:', await currentModel.textContent());
    console.log('Llama status:', await llamaStatus.textContent());
  });

  test('SSE connection is established', async ({ page }) => {
    // Listen for SSE connection
    let sseConnected = false;
    
    page.on('request', request => {
      if (request.url().includes('/events')) {
        sseConnected = true;
        console.log('SSE request made to /events');
      }
    });
    
    await page.goto('/');
    
    // Wait a moment for SSE to connect
    await page.waitForTimeout(1000);
    
    expect(sseConnected).toBe(true);
  });

  test('SSE receives initial status', async ({ page }) => {
    // Capture console messages to see SSE data
    const consoleMessages = [];
    page.on('console', msg => {
      consoleMessages.push(msg.text());
    });
    
    // Add a script to log SSE messages
    await page.addInitScript(() => {
      const originalEventSource = window.EventSource;
      // @ts-ignore
      window.EventSource = function(url) {
        console.log('EventSource connecting to:', url);
        const es = new originalEventSource(url);
        es.addEventListener('message', (e) => {
          console.log('SSE message received:', e.data);
        });
        return es;
      };
    });
    
    await page.goto('/');
    
    // Wait for SSE message
    await page.waitForTimeout(2000);
    
    // Check if we received an SSE message
    const sseMessages = consoleMessages.filter(m => m.includes('SSE message received'));
    console.log('SSE messages:', sseMessages);
    
    expect(sseMessages.length).toBeGreaterThan(0);
  });

  test('status bar updates when model switch is triggered via API', async ({ page, request }) => {
    await page.goto('/');
    
    // Wait for SSE connection to be established
    await page.waitForTimeout(1000);
    
    // Get initial model
    const currentModel = page.locator('#currentModelStatus');
    const llamaStatus = page.locator('#llamaServerStatus');
    const statusMessage = page.locator('#statusMessage');
    
    const initialModel = await currentModel.textContent();
    console.log('Initial model:', initialModel);
    
    // Determine which model to switch to
    // Updated defaults: prefer switching between qwen3 and gpt120 for the test.
    // If the initial model is neither, fall back to qwen3.
    const alt1 = 'qwen3';
    const alt2 = 'gpt120';
    const targetModel = initialModel === alt1 ? alt2 : alt1;
    console.log('Switching to:', targetModel);
    
    // Set up a promise to detect SSE switching event
    const switchingDetected = page.evaluate(() => {
      return new Promise((resolve) => {
        const checkInterval = setInterval(() => {
          const modelEl = document.getElementById('currentModelStatus');
          if (modelEl && modelEl.textContent.includes('Switching')) {
            clearInterval(checkInterval);
            resolve(true);
          }
        }, 100);
        // Timeout after 10 seconds
        setTimeout(() => {
          clearInterval(checkInterval);
          resolve(false);
        }, 10000);
      });
    });
    
    // Trigger model switch via API (this simulates an external request)
    // Don't await - let it run in background
    const switchPromise = request.post(`/admin/switch-model/${targetModel}`);
    
    // Wait for the switching state to be detected
    const detected = await switchingDetected;
    console.log('Switching state detected:', detected);
    
    if (detected) {
      // Verify the status bar shows switching state
      const currentModelText = await currentModel.textContent();
      const llamaStatusText = await llamaStatus.textContent();
      console.log('Current model during switch:', currentModelText);
      console.log('Llama status during switch:', llamaStatusText);
      
      expect(currentModelText).toContain('Switching');
      expect(llamaStatusText).toBe('Switching');
      
      // Check toast message
      const toastVisible = await statusMessage.isVisible();
      if (toastVisible) {
        console.log('Toast message:', await statusMessage.textContent());
      }
    }
    
    // Wait for switch to complete
    const response = await switchPromise;
    expect(response.ok()).toBe(true);
    
    // Status should update to show new model
    await expect(currentModel).toHaveText(targetModel, { timeout: 60000 });
    await expect(llamaStatus).toHaveText('Running', { timeout: 60000 });
    
    // The switching detection is the key assertion
    expect(detected).toBe(true);
  });

  test('Load Model button shows switching status', async ({ page }) => {
    await page.goto('/');
    
    const currentModel = page.locator('#currentModelStatus');
    const llamaStatus = page.locator('#llamaServerStatus');
    const statusMessage = page.locator('#statusMessage');
    
    const initialModel = await currentModel.textContent();
    console.log('Initial model:', initialModel);
    
    // Find a Load Model button (for a model that's not currently loaded)
    const loadButton = page.locator('button.btn-switch').first();
    
    if (await loadButton.count() === 0) {
      console.log('No Load Model button found - only one local model configured or current model is the only one');
      test.skip();
      return;
    }
    
    // Get the model name from the button text
    const buttonText = await loadButton.textContent();
    console.log('Button text:', buttonText);
    
    // Click the button
    await loadButton.click();
    
    // Check that switching status appears
    await expect(currentModel).toContainText('Switching', { timeout: 5000 });
    await expect(llamaStatus).toHaveText('Switching', { timeout: 5000 });
    await expect(statusMessage).toBeVisible({ timeout: 5000 });
    await expect(statusMessage).toContainText('Switching model to', { timeout: 5000 });
    
    console.log('Toast during switch:', await statusMessage.textContent());
    
    // Wait for completion (this can take a while)
    await expect(llamaStatus).toHaveText('Running', { timeout: 120000 });
  });

});

test.describe('API Passthrough Tests', () => {
  
  test('test endpoint shows switching status when model differs', async ({ page }) => {
    await page.goto('/');
    
    const currentModel = page.locator('#currentModelStatus');
    const llamaStatus = page.locator('#llamaServerStatus');
    const statusMessage = page.locator('#statusMessage');
    const modelSelect = page.locator('#modelSelect');
    
    const initialModel = await currentModel.textContent();
    console.log('Initial model:', initialModel);
    
    // Select a different local model in the dropdown
    const options = await modelSelect.locator('option').all();
    let targetModel = null;
    
    for (const option of options) {
      const value = await option.getAttribute('value');
      const text = await option.textContent();
      if (value !== initialModel && text?.includes('Local')) {
        targetModel = value;
        break;
      }
    }
    
    if (!targetModel) {
      console.log('No other local model to switch to');
      test.skip();
      return;
    }
    
    console.log('Selecting model:', targetModel);
    await modelSelect.selectOption(targetModel);
    
    // Click the Chat test button
    const chatTestButton = page.locator('button:has-text("Test")').first();
    await chatTestButton.click();
    
    // Check that switching status appears via SSE
    await expect(currentModel).toContainText('Switching', { timeout: 10000 });
    await expect(llamaStatus).toHaveText('Switching', { timeout: 10000 });
    await expect(statusMessage).toBeVisible({ timeout: 10000 });
    
    console.log('Toast during API test switch:', await statusMessage.textContent());
  });

});
