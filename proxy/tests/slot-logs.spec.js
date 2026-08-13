// @ts-check
const { test, expect } = require('@playwright/test');

/**
 * Install a URL-aware EventSource stub before the page loads.
 *
 * The page creates several EventSources with different URLs (/events and
 * per-slot /logs/tail streams).  routeTable entries are JSON-serializable
 * (strings only) so the whole table can cross the addInitScript boundary:
 *   { urlContains: string, payloads: object[], delayMs: number }
 * Every EventSource whose URL contains urlContains emits its payloads.
 * Unmatched URLs (e.g. the raw proxy/llama panes) stay quiet.
 */
function installFakeEventSource(routeTable) {
  window.EventSource = function windowEventSourceStub(url) {
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
    for (const route of routeTable) {
      if (url.indexOf(route.urlContains) !== -1) {
        route.payloads.forEach((payload, idx) => {
          setTimeout(() => {
            es.dispatchEvent(
              new MessageEvent('message', { data: JSON.stringify(payload) })
            );
          }, route.delayMs + idx * 50);
        });
      }
    }
    return es;
  };
}

const SESSION_A = '11111111-2222-3333-4444-555555555555';
const SESSION_B = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee';

function statusPayload(slots) {
  return {
    type: 'status',
    slots,
    llama_server_running: true,
    current_model: 'test-model',
    n_ctx: 4096,
    kv_cache_tokens: 128,
    total_sent: 0,
    total_recv: 0,
    per_model_queries: {},
  };
}

function twoSlotStatus() {
  return statusPayload([
    {
      slot_id: 2,
      is_processing: true,
      n_decoded: 100,
      n_tokens: 100,
      progress: 0.5,
      total_tokens: 200,
      session_id: SESSION_A,
      generation_done: false,
    },
    {
      slot_id: 3,
      is_processing: true,
      n_decoded: 50,
      n_tokens: 50,
      progress: 0.25,
      total_tokens: 200,
      session_id: SESSION_B,
      generation_done: false,
    },
  ]);
}

const eventsRoute = (payloads) => ({ urlContains: '/events', payloads, delayMs: 100 });

test.describe('Slot Logs Tab', () => {

  test('Slots tab is the default tab and renders one section per active slot', async ({ page }) => {
    await page.addInitScript(installFakeEventSource, [eventsRoute([twoSlotStatus()])]);

    await page.goto('/logs');

    // Default tab is Slots
    await expect(page.locator('#tabSlots')).toHaveClass(/active/);
    await expect(page.locator('#tabSlotsContent')).toBeVisible();
    await expect(page.locator('#tabAllLogsContent')).toBeHidden();

    // One section per active slot, with header (slot id, session, status)
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(2);
    const slot2 = page.locator('.slot-section[data-slot-id="2"]');
    await expect(slot2.locator('.slot-id')).toContainText('Slot 2');
    await expect(slot2.locator('.slot-session')).toContainText(SESSION_A.substring(0, 8));
    await expect(slot2.locator('.slot-status-badge')).toContainText('Processed');
  });

  test('All Logs tab keeps the two-pane log view and session dropdown', async ({ page }) => {
    await page.addInitScript(installFakeEventSource, [eventsRoute([twoSlotStatus()])]);

    await page.goto('/logs');

    await page.click('#tabAllLogs');
    await expect(page.locator('#tabAllLogsContent')).toBeVisible();
    await expect(page.locator('#tabSlotsContent')).toBeHidden();
    await expect(page.locator('#proxyPane')).toBeVisible();
    await expect(page.locator('#llamaPane')).toBeVisible();
    await expect(page.locator('#sessionSelect')).toBeVisible();

    // Switch back to Slots
    await page.click('#tabSlots');
    await expect(page.locator('#tabSlotsContent')).toBeVisible();
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(2);
  });

  test('slot sections share equal heights and are never squeezed below visibility', async ({ page }) => {
    await page.addInitScript(installFakeEventSource, [eventsRoute([twoSlotStatus()])]);

    await page.goto('/logs');
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(2);

    const box2 = await page.locator('.slot-section[data-slot-id="2"]').boundingBox();
    const box3 = await page.locator('.slot-section[data-slot-id="3"]').boundingBox();
    expect(box2).not.toBeNull();
    expect(box3).not.toBeNull();
    // Equal heights (within a small tolerance)
    expect(Math.abs(box2.height - box3.height)).toBeLessThanOrEqual(5);
    // At least 4 log lines visible: section min-height 150px (header ~38px
    // leaves a ~112px pane ≈ 5 lines at 0.85rem/1.6 line-height).
    expect(box2.height).toBeGreaterThanOrEqual(150);
    expect(box3.height).toBeGreaterThanOrEqual(150);
  });

  test('clicking a slot section expands it to full height; clicking again restores equal heights', async ({ page }) => {
    await page.addInitScript(installFakeEventSource, [eventsRoute([twoSlotStatus()])]);

    await page.goto('/logs');
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(2);

    const slot2 = page.locator('.slot-section[data-slot-id="2"]');
    const slot3 = page.locator('.slot-section[data-slot-id="3"]');

    // Expand slot 2
    await slot2.locator('.slot-section-header').click();
    await expect(slot2).toHaveClass(/expanded/);
    const expandedBox = await slot2.boundingBox();
    const collapsedBox = await slot3.boundingBox();
    expect(expandedBox.height).toBeGreaterThan(collapsedBox.height * 1.5);
    // The other slot stays visible (min-height)
    expect(collapsedBox.height).toBeGreaterThanOrEqual(150);

    // Collapse back to equal heights
    await slot2.locator('.slot-section-header').click();
    await expect(slot2).not.toHaveClass(/expanded/);
    const afterBox2 = await slot2.boundingBox();
    const afterBox3 = await slot3.boundingBox();
    expect(Math.abs(afterBox2.height - afterBox3.height)).toBeLessThanOrEqual(5);
  });

  test('per-slot streams are routed to their own section only', async ({ page }) => {
    const routes = [
      eventsRoute([twoSlotStatus()]),
      {
        urlContains: 'source=llama&slot=2',
        delayMs: 200,
        payloads: [
          { initial: '[57463] slot update_slots: id  2 | task 209403 | n_tokens = 16750, ...\n[57463] slot      release: id  2 | task 209403', source: 'llama', slot: 2 },
          { line: '[57463] slot print_timing: id  2 | task 209403', source: 'llama', slot: 2 },
        ],
      },
      {
        urlContains: 'source=proxy&slot=2',
        delayMs: 200,
        payloads: [
          { line: 'slot_save success session=' + SESSION_A + ' slot=2', source: 'proxy', slot: 2 },
        ],
      },
      {
        urlContains: 'source=llama&slot=3',
        delayMs: 200,
        payloads: [
          { line: '[57463] slot update_slots: id  3 | task 209410 | n_tokens = 9000, ...', source: 'llama', slot: 3 },
        ],
      },
    ];
    await page.addInitScript(installFakeEventSource, routes);

    await page.goto('/logs');
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(2);

    const slot2Log = page.locator('.slot-section[data-slot-id="2"] .slot-log');
    const slot3Log = page.locator('.slot-section[data-slot-id="3"] .slot-log');

    // Slot 2 receives its llama + proxy lines with source badges
    await expect(slot2Log.locator('.line')).toHaveCount(4);
    await expect(slot2Log).toContainText('id  2');
    await expect(slot2Log).toContainText('slot_save success');
    await expect(slot2Log.locator('.src-badge.llama')).toHaveCount(3);
    await expect(slot2Log.locator('.src-badge.proxy')).toHaveCount(1);

    // Slot 3 receives only its own lines — never slot 2's
    await expect(slot3Log.locator('.line')).toHaveCount(1);
    await expect(slot3Log).toContainText('id  3');
    await expect(slot3Log).not.toContainText('id  2');
    await expect(slot3Log).not.toContainText('slot_save success');
  });

  test('slot sections refresh as slots become active/inactive', async ({ page }) => {
    // Three /events payloads spaced wide enough for Playwright's polling to
    // observe each intermediate state (slot 5 joins, then slots 3+5 leave).
    const routes = [
      { urlContains: '/events', delayMs: 100, payloads: [twoSlotStatus()] },
      {
        urlContains: '/events',
        delayMs: 800,
        payloads: [
          statusPayload([
            ...twoSlotStatus().slots,
            { slot_id: 5, is_processing: false, n_decoded: 10, session_id: 'cccccccc-1111-2222-3333-444444444444', generation_done: true },
          ]),
        ],
      },
      {
        urlContains: '/events',
        delayMs: 1600,
        payloads: [
          statusPayload([
            { slot_id: 2, is_processing: true, n_decoded: 100, n_tokens: 100, progress: 0.5, total_tokens: 200, session_id: SESSION_A, generation_done: false },
          ]),
        ],
      },
    ];
    await page.addInitScript(installFakeEventSource, routes);

    await page.goto('/logs');

    // Initial: slots 2 + 3
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(2);

    // Slot 5 joins (recently-finished session -> still active)
    await expect(page.locator('.slot-section[data-slot-id="5"]')).toHaveCount(1);
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(3);

    // Slots 3 and 5 leave -> only slot 2 remains
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(1);
    await expect(page.locator('.slot-section[data-slot-id="2"]')).toHaveCount(1);
  });

  test('slot sections are ordered numerically by slot id regardless of payload order', async ({ page }) => {
    // Regression for the manual-review rejection (2026-08-13): llama-server
    // /slots can return slots in non-numeric order (and the shared cache may
    // reshuffle); sections must render in numeric order (0, 1, 2, ...).
    const shuffled = statusPayload([
      { slot_id: 2, is_processing: true, n_decoded: 10, n_tokens: 10, progress: 0.05, total_tokens: 200, session_id: SESSION_B, generation_done: false },
      { slot_id: 0, is_processing: false, n_decoded: 0, generation_done: false },
      { slot_id: 3, is_processing: false, n_decoded: 0, generation_done: false },
      { slot_id: 1, is_processing: false, n_decoded: 0, generation_done: false },
    ]);
    await page.addInitScript(installFakeEventSource, [eventsRoute([shuffled])]);

    await page.goto('/logs');
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(4);

    const ids = await page.$$eval('#slotSections .slot-section', (secs) =>
      secs.map((s) => Number(s.dataset.slotId))
    );
    expect(ids).toEqual([0, 1, 2, 3]);
  });

  test('a slot joining mid-range keeps the numeric DOM order', async ({ page }) => {
    // Initial payload is missing slot 1 (sections 0 and 2 render first); a
    // later payload adds slot 1. The new section must slot into position,
    // not be appended at the end ([0,2,1] would be wrong).
    const first = statusPayload([
      { slot_id: 0, is_processing: false, n_decoded: 0, generation_done: false },
      { slot_id: 2, is_processing: false, n_decoded: 0, generation_done: false },
    ]);
    const second = statusPayload([
      { slot_id: 0, is_processing: false, n_decoded: 0, generation_done: false },
      { slot_id: 1, is_processing: false, n_decoded: 0, generation_done: false },
      { slot_id: 2, is_processing: false, n_decoded: 0, generation_done: false },
    ]);
    const routes = [
      { urlContains: '/events', delayMs: 100, payloads: [first] },
      { urlContains: '/events', delayMs: 700, payloads: [second] },
    ];
    await page.addInitScript(installFakeEventSource, routes);

    await page.goto('/logs');
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(2);

    // Slot 1 joins mid-range; DOM order must stay numeric
    await expect(page.locator('.slot-section[data-slot-id="1"]')).toHaveCount(1);
    const ids = await page.$$eval('#slotSections .slot-section', (secs) =>
      secs.map((s) => Number(s.dataset.slotId))
    );
    expect(ids).toEqual([0, 1, 2]);
  });

  test('a working slot with no log lines yet shows a status placeholder, cleared when lines arrive', async ({ page }) => {
    // Regression for the manual-review rejection (2026-08-13): during
    // pre-fill/working periods llama-server has not emitted slot log lines
    // yet, so the pane must not look empty — show a status-aware placeholder.
    const routes = [
      {
        urlContains: '/events',
        delayMs: 100,
        payloads: [
          statusPayload([
            { slot_id: 1, is_processing: true, n_decoded: 0, session_id: SESSION_A, generation_done: false },
          ]),
        ],
      },
      {
        urlContains: 'source=llama&slot=1',
        delayMs: 400,
        payloads: [
          { line: '[57463] slot update_slots: id  1 | task 209403 | n_tokens = 16750, ...', source: 'llama', slot: 1 },
        ],
      },
    ];
    await page.addInitScript(installFakeEventSource, routes);

    await page.goto('/logs');
    const section = page.locator('.slot-section[data-slot-id="1"]');
    const logEl = section.locator('.slot-log');

    // Before any line arrives, a placeholder is visible (never an empty pane)
    await expect(logEl.locator('.slot-empty-hint')).toBeVisible();
    await expect(logEl.locator('.slot-empty-hint')).toContainText(/no log lines|waiting|pre-fill/i);
    await expect(logEl.locator('.line')).toHaveCount(0);

    // Once the first relevant line arrives, the placeholder is removed
    await expect(logEl.locator('.line')).toHaveCount(1);
    await expect(logEl.locator('.slot-empty-hint')).toHaveCount(0);
  });

  test('idle slots are still rendered so the Slots tab is never empty', async ({ page }) => {
    // Regression for the manual-review rejection (2026-08-10): when no slot
    // is processing and no dispatch session is mapped yet, every slot must
    // still appear with an Idle badge — mirroring the home-page slot cards.
    const idlePayload = statusPayload([
      { slot_id: 0, is_processing: false, n_decoded: 0, generation_done: false },
      { slot_id: 1, is_processing: false, n_decoded: 0, generation_done: false },
    ]);
    await page.addInitScript(installFakeEventSource, [eventsRoute([idlePayload])]);

    await page.goto('/logs');
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(2);
    await expect(page.locator('.slot-section[data-slot-id="0"] .slot-status-badge')).toContainText('Idle');
    await expect(page.locator('.slot-section[data-slot-id="1"] .slot-status-badge')).toContainText('Idle');
    await expect(page.locator('.slot-section[data-slot-id="0"] .slot-session')).toContainText('no session yet');
    await expect(page.locator('#slotSummary')).toContainText('all idle');

    // A later update marks slot 1 processing -> badge flips live, section stays
    const processingPayload = statusPayload([
      { slot_id: 0, is_processing: false, n_decoded: 0, generation_done: false },
      { slot_id: 1, is_processing: true, n_decoded: 42, n_tokens: 42, progress: 0.2, total_tokens: 210, session_id: SESSION_B, generation_done: false },
    ]);
    const routes = [
      { urlContains: '/events', delayMs: 100, payloads: [idlePayload] },
      { urlContains: '/events', delayMs: 700, payloads: [processingPayload] },
    ];
    await page.addInitScript(installFakeEventSource, routes);
    await page.goto('/logs');
    await expect(page.locator('#slotSections .slot-section')).toHaveCount(2);
    await expect(page.locator('.slot-section[data-slot-id="1"] .slot-status-badge')).toContainText('Processed');
  });
});
