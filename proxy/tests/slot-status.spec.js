// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Build an EventSource stub that only emits the given payload(s) and never
 * connects to the real /events stream. This makes UI assertions deterministic
 * instead of racing with the live proxy's real SSE broadcasts.
 *
 * The page's connectSSE() assigns `eventSource.onmessage` after construction,
 * so the stub must invoke that handler (and any addEventListener('message')
 * callbacks) when a synthetic event is dispatched.
 */
function fakeEventSource(payloads, delayMs) {
  return function windowEventSourceStub() {
    const listeners = { message: [] };
    const es = {
      onmessage: null,
      addEventListener(type, cb) {
        (listeners[type] || (listeners[type] = [])).push(cb);
      },
      dispatchEvent(evt) {
        if (evt.type === 'message') {
          if (es.onmessage) es.onmessage.call(es, evt);
          (listeners.message || []).forEach((cb) => cb.call(es, evt));
        }
        return true;
      },
      close() {},
    };
    payloads.forEach((payload, idx) => {
      setTimeout(() => {
        es.dispatchEvent(
          new MessageEvent('message', { data: JSON.stringify(payload) })
        );
      }, delayMs + idx * 50);
    });
    return es;
  };
}

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
    // Deterministic fake EventSource: only the synthetic all-idle payload is
    // seen by the UI (no real /events stream to race with). The inlined stub
    // is serializable into the browser context and matches the sibling
    // deterministic tests in this spec (LP-0MSHWZPHY0058NRA). A pure-idle
    // payload keeps this distinct from 'slot cards show correct status
    // colors', which covers the mixed idle/processing states.
    await page.addInitScript(({ payload, delayMs }) => {
      const listeners = { message: [] };
      window.EventSource = function () {
        const es = {
          onmessage: null,
          addEventListener(type, cb) {
            (listeners[type] || (listeners[type] = [])).push(cb);
          },
          dispatchEvent(evt) {
            if (evt.type === 'message') {
              if (es.onmessage) es.onmessage.call(es, evt);
              (listeners.message || []).forEach((cb) => cb.call(es, evt));
            }
            return true;
          },
          close() {},
        };
        setTimeout(() => {
          es.dispatchEvent(
            new MessageEvent('message', { data: JSON.stringify(payload) })
          );
        }, delayMs);
        return es;
      };
    }, {
      payload: {
        type: 'status',
        slots: [
          { slot_id: 0, is_processing: false },
          { slot_id: 1, is_processing: false }
        ],
        llama_server_running: true,
        current_model: 'test-model',
        n_ctx: 4096,
        kv_cache_tokens: 128,
        total_sent: 0,
        total_recv: 0,
        per_model_queries: {}
      },
      delayMs: 100,
    });

    await page.goto('/');

    // Slot cards are rendered from the SSE payload
    const slotCards = page.locator('.slot-card');
    await expect(slotCards).toHaveCount(2);

    // Each slot card shows its identifier and an Idle status badge
    for (let i = 0; i < 2; i++) {
      const card = slotCards.nth(i);
      await expect(card.locator('.slot-id')).toBeVisible();
      const statusBadge = card.locator('.slot-status-badge');
      await expect(statusBadge).toBeVisible();
      await expect(statusBadge).toHaveText('Idle');
    }
  });

  test('SSE payload includes slots field', async ({ page }) => {
    // Override BEFORE navigation so the page's EventSource is wrapped from
    // the start; addEventListener-based capture survives the page assigning
    // its own onmessage handler (LP-0MSGC26BM006DS6M).
    await page.addInitScript(() => {
      const originalEventSource = window.EventSource;
      window.EventSource = function (url) {
        const es = new originalEventSource(url);
        es.addEventListener('message', function (event) {
          window.lastSSEMessage = event.data;
        });
        return es;
      };
    });

    await page.goto('/');

    // Real status broadcasts include a slots field; wait for the first one
    // instead of a blind sleep (broadcasts fire on status events).
    await page.waitForFunction(
      () => window.lastSSEMessage !== undefined,
      null,
      { timeout: 30000 }
    );

    const message = await page.evaluate(() => window.lastSSEMessage);
    expect(message).toBeDefined();

    const parsed = JSON.parse(message);
    expect(parsed).toHaveProperty('slots');
  });

  test('slot status updates when SSE delivers new data', async ({ page }) => {
    // Collect SSE messages via addEventListener (survives the page's own
    // onmessage assignment) (LP-0MSGC26C4000W7XD).
    await page.addInitScript(() => {
      const originalEventSource = window.EventSource;
      window.EventSource = function (url) {
        const es = new originalEventSource(url);
        es.addEventListener('message', function (event) {
          if (!window.__sseMessages) window.__sseMessages = [];
          window.__sseMessages.push(event.data);
        });
        return es;
      };
    });

    await page.goto('/');

    // Wait for at least one real SSE message (no blind sleep).
    await page.waitForFunction(
      () => (window.__sseMessages || []).length > 0,
      null,
      { timeout: 30000 }
    );

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
    // Deterministic fake EventSource: only the empty-slots payload is seen by
    // the UI (no real stream to race with). The stub is inlined so it is
    // serializable into the browser context (LP-0MSGC2CS00022Y2W).
    await page.addInitScript(({ payload, delayMs }) => {
      const listeners = { message: [] };
      window.EventSource = function () {
        const es = {
          onmessage: null,
          addEventListener(type, cb) {
            (listeners[type] || (listeners[type] = [])).push(cb);
          },
          dispatchEvent(evt) {
            if (evt.type === 'message') {
              if (es.onmessage) es.onmessage.call(es, evt);
              (listeners.message || []).forEach((cb) => cb.call(es, evt));
            }
            return true;
          },
          close() {},
        };
        setTimeout(() => {
          es.dispatchEvent(
            new MessageEvent('message', { data: JSON.stringify(payload) })
          );
        }, delayMs);
        return es;
      };
    }, {
      payload: {
        type: 'status',
        slots: [],
        llama_server_running: true,
        current_model: 'test-model',
        n_ctx: 4096,
        kv_cache_tokens: 128,
        total_sent: 0,
        total_recv: 0,
        per_model_queries: {}
      },
      delayMs: 100,
    });

    await page.goto('/');

    // Should show empty state
    const emptyState = page.locator('#slotStatusEmpty');
    await expect(emptyState).toBeVisible();
    await expect(emptyState).toContainText('No slot data');
  });

  test('slot cards show correct status colors', async ({ page }) => {
    // Deterministic fake EventSource with mixed slot states matching the
    // current template schema (n_tokens/progress, not n_decoded). Inlined
    // stub (serializable into the browser context).
    await page.addInitScript(({ payload, delayMs }) => {
      const listeners = { message: [] };
      window.EventSource = function () {
        const es = {
          onmessage: null,
          addEventListener(type, cb) {
            (listeners[type] || (listeners[type] = [])).push(cb);
          },
          dispatchEvent(evt) {
            if (evt.type === 'message') {
              if (es.onmessage) es.onmessage.call(es, evt);
              (listeners.message || []).forEach((cb) => cb.call(es, evt));
            }
            return true;
          },
          close() {},
        };
        setTimeout(() => {
          es.dispatchEvent(
            new MessageEvent('message', { data: JSON.stringify(payload) })
          );
        }, delayMs);
        return es;
      };
    }, {
      payload: {
        type: 'status',
        slots: [
          { slot_id: 0, is_processing: false },
          { slot_id: 1, is_processing: true, n_tokens: 42, progress: 0.3, total_tokens: 140 },
          { slot_id: 2, is_processing: true, n_tokens: 100, progress: 0.5, total_tokens: 200 }
        ],
        llama_server_running: true,
        current_model: 'test-model',
        n_ctx: 4096,
        kv_cache_tokens: 128,
        total_sent: 0,
        total_recv: 0,
        per_model_queries: {}
      },
      delayMs: 100,
    });

    await page.goto('/');

    // Should have 3 slot cards
    const slotCards = page.locator('.slot-card');
    await expect(slotCards).toHaveCount(3);

    // First card should be idle
    const firstStatus = slotCards.nth(0).locator('.slot-status-badge');
    await expect(firstStatus).toHaveText('Idle');

    // Second card should show processing with token count
    const secondStatus = slotCards.nth(1).locator('.slot-status-badge');
    await expect(secondStatus).toContainText('Processed');
    await expect(secondStatus).toContainText('42');

    // Third card should show processing with 100 tokens
    const thirdStatus = slotCards.nth(2).locator('.slot-status-badge');
    await expect(thirdStatus).toContainText('Processed');
    await expect(thirdStatus).toContainText('100');
  });

  test('slot identifier is displayed per card', async ({ page }) => {
    // Deterministic fake EventSource with two slots (inlined stub).
    await page.addInitScript(({ payload, delayMs }) => {
      const listeners = { message: [] };
      window.EventSource = function () {
        const es = {
          onmessage: null,
          addEventListener(type, cb) {
            (listeners[type] || (listeners[type] = [])).push(cb);
          },
          dispatchEvent(evt) {
            if (evt.type === 'message') {
              if (es.onmessage) es.onmessage.call(es, evt);
              (listeners.message || []).forEach((cb) => cb.call(es, evt));
            }
            return true;
          },
          close() {},
        };
        setTimeout(() => {
          es.dispatchEvent(
            new MessageEvent('message', { data: JSON.stringify(payload) })
          );
        }, delayMs);
        return es;
      };
    }, {
      payload: {
        type: 'status',
        slots: [
          { slot_id: 0, is_processing: false },
          { slot_id: 1, is_processing: true, n_tokens: 42, progress: 0.3, total_tokens: 140 }
        ],
        llama_server_running: true,
        current_model: 'test-model',
        n_ctx: 4096,
        kv_cache_tokens: 128,
        total_sent: 0,
        total_recv: 0,
        per_model_queries: {}
      },
      delayMs: 100,
    });

    await page.goto('/');

    const slotCards = page.locator('.slot-card');
    await expect(slotCards).toHaveCount(2);

    await expect(slotCards.nth(0).locator('.slot-id')).toContainText('Slot 0');
    await expect(slotCards.nth(1).locator('.slot-id')).toContainText('Slot 1');
  });

});
