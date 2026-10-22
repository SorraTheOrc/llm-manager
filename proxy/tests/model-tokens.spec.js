// @ts-check
const { test, expect } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

/**
 * Per-model token aggregation in the /logs summary header.
 *
 * Parent feature: LP-0MVE592AU008VAP4. Test item: LP-0MVEUXJ83005I880.
 *
 * These specs render the repository's own `view_logs.html` template with an
 * injected `window.__INITIAL_STATS` snapshot, so the assertions always target
 * the template under test rather than whatever the running proxy happens to
 * have deployed. All other network traffic is stubbed (`EventSource` is
 * silenced and `fetch` returns empty JSON) so the snapshot is the only input
 * to `renderSummary()` and the suite is deterministic and CI-safe.
 *
 * DOM contract asserted here (implemented by LP-0MVEUXLHZ0054LGX):
 *
 *   <div id="modelTokens">
 *     <div class="model-token-row" data-model="<name>"
 *          data-sent="<int>" data-recv="<int>"
 *          data-user="<int>" data-assistant="<int>" data-tool="<int>"
 *          data-system="<int>" data-response="<int>">
 *       <!-- visible name + sent/recv totals + category breakdown -->
 *     </div>
 *     ...
 *   </div>
 *
 * Aggregation rules exercised by the fixture:
 *   - `-> model:<name>` suffixes and bare `model:<name>` keys group under
 *     that model (both sent and recv).
 *   - Different endpoint paths that share a model name aggregate into one row.
 *   - Keys with no model suffix group under the `unknown` row.
 *   - The global `total_sent` / `total_recv` keys belong to no model and must
 *     not be attributed to `unknown`.
 */

const TEMPLATE_PATH = path.join(__dirname, '..', 'templates', 'view_logs.html');

const COUNTS = {
  'POST /v1/chat/completions -> model:qwen3.6': 5,
  'POST /v1/models -> local': 1,
};

/**
 * Token-count fixture. Numbers are chosen so every expected aggregate is
 * unambiguous (see the per-test assertions).
 */
const TOKENS = {
  // qwen3.6 on two different endpoints + bare model counters.
  'sent:user:POST /v1/chat/completions -> model:qwen3.6': 1000,
  'sent:assistant:POST /v1/chat/completions -> model:qwen3.6': 200,
  'sent:tool:POST /v1/chat/completions -> model:qwen3.6': 300,
  'sent:system:POST /v1/chat/completions -> model:qwen3.6': 400,
  'recv:response:POST /v1/chat/completions -> model:qwen3.6': 500,
  'sent:user:POST /v1/completions -> model:qwen3.6': 60,
  'sent:model:qwen3.6': 50,
  'recv:model:qwen3.6': 25,

  // gpt-oss: category keys plus a bare model counter.
  'sent:user:POST /v1/completions -> model:gpt-oss': 700,
  'recv:response:POST /v1/completions -> model:gpt-oss': 900,
  'sent:model:gpt-oss': 10,

  // Legacy keys without any model suffix -> "unknown".
  'sent:user:POST /v1/embeddings -> local': 111,
  'recv:response:POST /v1/embeddings -> local': 222,
  'sent:POST /v1/models -> local': 5,
  'recv:POST /v1/models -> local': 7,

  // Global totals: never attributed to a model (nor to "unknown").
  total_sent: 123456789,
  total_recv: 987654321,
};

/** Expected per-model aggregates derived from TOKENS. */
const EXPECTED_MODELS = {
  'qwen3.6': {
    sent: 2010, recv: 525,
    user: 1060, assistant: 200, tool: 300, system: 400, response: 500,
  },
  'gpt-oss': {
    sent: 710, recv: 900,
    user: 700, assistant: 0, tool: 0, system: 0, response: 900,
  },
  unknown: {
    sent: 116, recv: 229,
    user: 111, assistant: 0, tool: 0, system: 0, response: 222,
  },
};

/** Render the repository template with an injected initial-stats snapshot. */
function renderTemplate() {
  let html = fs.readFileSync(TEMPLATE_PATH, 'utf8');
  html = html.replace('__LLAMA_SERVER_VERSION__', 'test-llama');
  html = html.replace('__ROCM_VERSION__', 'test-rocm');
  html = html.replace('__LOG_VIEWER_SCRIPT__', '');
  const statsScript =
    '<script>window.__INITIAL_STATS = ' +
    JSON.stringify({ counts: COUNTS, tokens: TOKENS }) +
    ';</script>';
  // Inject before the page's main script so renderSummary() sees the snapshot
  // on its initial run (the server normally injects it at the end of <body>).
  html = html.replace('<body>', '<body>' + statsScript);
  return html;
}

/**
 * Load /logs with the fixture snapshot and every other request stubbed.
 * EventSource is replaced by a silent stub so live SSE cannot overwrite the
 * injected snapshot after the first render.
 */
async function loadPage(page) {
  const html = renderTemplate();
  await page.addInitScript(() => {
    window.EventSource = function SilentEventSource() {
      return {
        onmessage: null,
        onerror: null,
        addEventListener() {},
        removeEventListener() {},
        close() {},
      };
    };
  });
  await page.route('**/*', (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    if (pathname === '/logs') {
      return route.fulfill({ status: 200, contentType: 'text/html', body: html });
    }
    if (request.resourceType() === 'document') {
      return route.fulfill({ status: 404, contentType: 'text/plain', body: 'not found' });
    }
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.goto('/logs');
  // The summary header (endpoint + per-model token sections) lives in the
  // "All Logs" tab, which is hidden by default — activate it.
  await page.locator('#tabAllLogs').click();
  await expect(page.locator('#modelTokens')).toBeVisible();
}

/** Numeric values rendered in a row, with thousands separators normalised. */
async function renderedNumbers(row) {
  const text = await row.innerText();
  return (text.match(/\d[\d,]*/g) || []).map((value) => value.replace(/,/g, ''));
}

test.describe('Per-model token summary (/logs)', () => {
  test('renders one row per distinct model with total sent and recv', async ({ page }) => {
    await loadPage(page);

    const rows = page.locator('#modelTokens .model-token-row');
    await expect(rows).toHaveCount(Object.keys(EXPECTED_MODELS).length);

    for (const [model, totals] of Object.entries(EXPECTED_MODELS)) {
      const row = page.locator(
        `#modelTokens .model-token-row[data-model="${model}"]`
      );
      await expect(row, `expected exactly one row for model ${model}`).toHaveCount(1);

      await expect(row).toHaveAttribute('data-sent', String(totals.sent));
      await expect(row).toHaveAttribute('data-recv', String(totals.recv));

      // The row's visible text must show the model name and both totals.
      await expect(row).toContainText(model);
      const numbers = await renderedNumbers(row);
      expect(numbers).toContain(String(totals.sent));
      expect(numbers).toContain(String(totals.recv));
    }
  });

  test('renders the per-model category breakdown', async ({ page }) => {
    await loadPage(page);

    for (const [model, totals] of Object.entries(EXPECTED_MODELS)) {
      const row = page.locator(
        `#modelTokens .model-token-row[data-model="${model}"]`
      );
      await expect(row).toHaveCount(1);

      await expect(row).toHaveAttribute('data-user', String(totals.user));
      await expect(row).toHaveAttribute('data-assistant', String(totals.assistant));
      await expect(row).toHaveAttribute('data-tool', String(totals.tool));
      await expect(row).toHaveAttribute('data-system', String(totals.system));
      await expect(row).toHaveAttribute('data-response', String(totals.response));
    }
  });

  test('groups keys without a model suffix under "unknown"', async ({ page }) => {
    await loadPage(page);

    const unknown = page.locator(
      '#modelTokens .model-token-row[data-model="unknown"]'
    );
    await expect(unknown).toHaveCount(1);
    await expect(unknown).toHaveAttribute('data-sent', '116');
    await expect(unknown).toHaveAttribute('data-recv', '229');
    await expect(unknown).toHaveAttribute('data-user', '111');
    await expect(unknown).toHaveAttribute('data-response', '222');

    // Global totals are not models: neither an "unknown" attribution nor a
    // phantom row of their own.
    await expect(
      page.locator('#modelTokens .model-token-row[data-model="total_sent"]')
    ).toHaveCount(0);
    await expect(
      page.locator('#modelTokens .model-token-row[data-model="total_recv"]')
    ).toHaveCount(0);
  });

  test('keeps the existing per-endpoint token section unchanged', async ({ page }) => {
    await loadPage(page);

    const endpointRows = page.locator('#tokens .line');
    await expect(endpointRows).toHaveCount(4);

    // Per-model rows live in their own container, never in the endpoint list.
    await expect(page.locator('#tokens .model-token-row')).toHaveCount(0);

    const chat = page.locator('#tokens .line', { hasText: 'Chat:' });
    await expect(chat).toHaveCount(1);
    await expect(chat).toContainText('User 1000');
    await expect(chat).toContainText('Asst. 200');
    await expect(chat).toContainText('Tool 300');
    await expect(chat).toContainText('Sys. 400');
    await expect(chat).toContainText('resp. 500');

    const completions = page.locator('#tokens .line', { hasText: 'Completions:' });
    await expect(completions).toHaveCount(1);
    // Note: the existing endpoint matcher matches paths by substring, so the
    // chat-completions keys also land under Completions (`/completions` is a
    // substring of `chat/completions`). These values ARE the pre-existing dev
    // baseline (verified on dev before this change) — the regression contract.
    await expect(completions).toContainText('User 1760');
    await expect(completions).toContainText('resp. 1400');
    await expect(completions).toContainText('Asst. 200');

    const embeddings = page.locator('#tokens .line', { hasText: 'Embeddings:' });
    await expect(embeddings).toHaveCount(1);
    await expect(embeddings).toContainText('User 111');
    await expect(embeddings).toContainText('resp. 222');

    // Flat (category-less) legacy keys still show on their endpoint.
    const models = page.locator('#tokens .line', { hasText: 'Models:' });
    await expect(models).toHaveCount(1);
    await expect(models).toContainText('sent 5');
    await expect(models).toContainText('recv 7');
  });
});
