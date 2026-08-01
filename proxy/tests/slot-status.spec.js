// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Slot Status Section', () => {
  
  test('slot status section exists on home tab', async ({ page }) => {
    await page.goto('/');
    
    const slotStatusSection = page.locator('#slotStatusSection');
    await expect(slotStatusSection).toBeVisible();
    
    const sectionTitle = page.locator('#slotStatusSection h2');
    await expect(sectionTitle).toContainText('Slot Status');
  });

  test('slot status section is above model endpoints table', async ({ page }) => {
    await page.goto('/');
    
    const slotSection = page.locator('#slotStatusSection');
    const modelTable = page.locator('#modelEndpointTable');
    
    // Check DOM order: slot section should come before model table
    const slotBox = await slotSection.boundingBox();
    const tableBox = await modelTable.boundingBox();
    
    if (slotBox && tableBox) {
      expect(slotBox.y + slotBox.height).toBeLessThanOrEqual(tableBox.y);
    }
  });

  test('shows idle slots from SSE data', async ({ page }) => {
    await page.goto('/');
    
    // Wait for SSE to deliver slot data
    await page.waitForTimeout(3000);
    
    // Check that slot cards are rendered
    const slotCards = page.locator('.slot-card');
    const cardCount = await slotCards.count();
    
    if (cardCount > 0) {
      // If there are slots, verify they have required elements
      const firstCard = slotCards.first();
      
      // Slot should have an identifier
      await expect(firstCard.locator('.slot-id')).toBeVisible();
      
      // Slot should have a status indicator
      const statusBadge = firstCard.locator('.slot-status-badge');
      await expect(statusBadge).toBeVisible();
      const statusText = await statusBadge.textContent();
      expect(['Idle', 'Processing']).toContain(statusText);
    }
  });

  test('SSE payload includes slots field', async ({ page }) => {
    const lastSSEMessage = await page.evaluate(() => {
      return new Promise((resolve) => {
        const originalEventSource = window.EventSource;
        window.EventSource = function(url) {
          const es = new originalEventSource(url);
          es.onmessage = function(event) {
            window.lastSSEMessage = event.data;
            resolve(event.data);
          };
          return es;
        };
      });
    });
    
    await page.goto('/');
    
    const message = await lastSSEMessage;
    expect(message).toBeDefined();
    
    const parsed = JSON.parse(message);
    expect(parsed).toHaveProperty('slots');
  });

  test('slot status updates when SSE delivers new data', async ({ page }) => {
    // Collect SSE messages for analysis
    const sseMessages = [];
    
    await page.addInitScript(() => {
      const originalEventSource = window.EventSource;
      window.EventSource = function(url) {
        const es = new originalEventSource(url);
        const originalOnMessage = es.onmessage;
        es.onmessage = function(event) {
          if (!window.__sseMessages) window.__sseMessages = [];
          window.__sseMessages.push(event.data);
          if (originalOnMessage) originalOnMessage.call(es, event);
        };
        return es;
      };
    });
    
    await page.goto('/');
    
    // Wait for several SSE messages
    await page.waitForTimeout(5000);
    
    const messages = await page.evaluate(() => window.__sseMessages || []);
    expect(messages.length).toBeGreaterThan(0);
    
    // At least one message should have a slots field
    const hasSlotsField = messages.some(msg => {
      try {
        const parsed = JSON.parse(msg);
        return parsed.slots !== undefined;
      } catch {
        return false;
      }
    });
    expect(hasSlotsField).toBe(true);
  });

  test('shows appropriate empty state when no slot data', async ({ page }) => {
    // Override EventSource to send a message with empty slots
    await page.addInitScript(() => {
      const originalEventSource = window.EventSource;
      window.EventSource = function(url) {
        const es = new originalEventSource(url);
        // Send a custom status event with empty slots after connection
        setTimeout(() => {
          const event = new MessageEvent('message', {
            data: JSON.stringify({
              type: 'status',
              slots: [],
              llama_server_running: true,
              current_model: 'test-model',
              n_ctx: 4096,
              kv_cache_tokens: 128,
              total_sent: 0,
              total_recv: 0,
              per_model_queries: {}
            })
          });
          es.dispatchEvent(event);
        }, 500);
        return es;
      };
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Should show empty state
    const emptyState = page.locator('#slotStatusEmpty');
    await expect(emptyState).toBeVisible();
    await expect(emptyState).toContainText('No slot data');
  });

  test('slot cards show correct status colors', async ({ page }) => {
    // Override EventSource to send a message with mixed slot states
    await page.addInitScript(() => {
      const originalEventSource = window.EventSource;
      window.EventSource = function(url) {
        const es = new originalEventSource(url);
        setTimeout(() => {
          const event = new MessageEvent('message', {
            data: JSON.stringify({
              type: 'status',
              slots: [
                { slot_id: 0, is_processing: false, n_decoded: null },
                { slot_id: 1, is_processing: true, n_decoded: 42 },
                { slot_id: 2, is_processing: true, n_decoded: 100 }
              ],
              llama_server_running: true,
              current_model: 'test-model',
              n_ctx: 4096,
              kv_cache_tokens: 128,
              total_sent: 0,
              total_recv: 0,
              per_model_queries: {}
            })
          });
          es.dispatchEvent(event);
        }, 500);
        return es;
      };
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Should have 3 slot cards
    const slotCards = page.locator('.slot-card');
    await expect(slotCards).toHaveCount(3);
    
    // First card should be idle (green)
    const firstStatus = slotCards.nth(0).locator('.slot-status-badge');
    await expect(firstStatus).toHaveText('Idle');
    
    // Second card should show processing with token count
    const secondStatus = slotCards.nth(1).locator('.slot-status-badge');
    await expect(secondStatus).toContainText('Processing');
    await expect(secondStatus).toContainText('42');
    
    // Third card should show processing with 100 tokens
    const thirdStatus = slotCards.nth(2).locator('.slot-status-badge');
    await expect(thirdStatus).toContainText('Processing');
    await expect(thirdStatus).toContainText('100');
  });

  test('slot identifier is displayed per card', async ({ page }) => {
    await page.addInitScript(() => {
      const originalEventSource = window.EventSource;
      window.EventSource = function(url) {
        const es = new originalEventSource(url);
        setTimeout(() => {
          const event = new MessageEvent('message', {
            data: JSON.stringify({
              type: 'status',
              slots: [
                { slot_id: 0, is_processing: false, n_decoded: null },
                { slot_id: 1, is_processing: true, n_decoded: 42 }
              ],
              llama_server_running: true,
              current_model: 'test-model',
              n_ctx: 4096,
              kv_cache_tokens: 128,
              total_sent: 0,
              total_recv: 0,
              per_model_queries: {}
            })
          });
          es.dispatchEvent(event);
        }, 500);
        return es;
      };
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    const slotCards = page.locator('.slot-card');
    await expect(slotCards).toHaveCount(2);
    
    await expect(slotCards.nth(0).locator('.slot-id')).toContainText('Slot 0');
    await expect(slotCards.nth(1).locator('.slot-id')).toContainText('Slot 1');
  });

});
