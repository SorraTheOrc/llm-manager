// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Diagnostic Tests', () => {
  
  test('check current page HTML structure', async ({ page }) => {
    await page.goto('/');
    
    // Get the status bar HTML
    const statusBar = page.locator('.status-bar');
    const statusBarHTML = await statusBar.innerHTML();
    console.log('Status bar HTML:', statusBarHTML);
    
    // Check for ID attributes
    const hasCurrentModelId = statusBarHTML.includes('id="currentModelStatus"');
    const hasLlamaStatusId = statusBarHTML.includes('id="llamaServerStatus"');
    
    console.log('Has currentModelStatus ID:', hasCurrentModelId);
    console.log('Has llamaServerStatus ID:', hasLlamaStatusId);
    
    if (!hasCurrentModelId || !hasLlamaStatusId) {
      console.log('ERROR: IDs are missing - server needs to be restarted with new code');
    }
    
    // Check for statusMessage element
    const statusMessage = page.locator('#statusMessage');
    const statusMessageExists = await statusMessage.count() > 0;
    console.log('Has statusMessage element:', statusMessageExists);
  });

  test('check SSE endpoint directly', async ({ page }) => {
    // /events is an infinite SSE stream; request.get() awaits the full body
    // and would never resolve. Instead, open the stream in the browser, read
    // the first chunk (proving the endpoint is live and SSE-formatted), then
    // abort the connection (LP-0MSGA6SKI0085UO6).
    await page.goto('/');
    const result = await page.evaluate(async () => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10000);
      try {
        const response = await fetch(`${location.origin}/events`, {
          headers: { 'Accept': 'text/event-stream' },
          signal: controller.signal,
        });
        const reader = response.body.getReader();
        const { value } = await reader.read();
        const firstChunk = value ? new TextDecoder().decode(value) : '';
        return {
          status: response.status,
          contentType: response.headers.get('content-type'),
          firstChunk,
        };
      } finally {
        clearTimeout(timeout);
        controller.abort();
      }
    });

    console.log('SSE endpoint status:', result.status);
    console.log('SSE content-type:', result.contentType);
    console.log('SSE first chunk:', JSON.stringify(result.firstChunk));

    // We can't easily read SSE with request API, but we can check it exists
    expect(result.status).toBe(200);
    expect(result.contentType || '').toContain('text/event-stream');
    expect(result.firstChunk.length).toBeGreaterThan(0);
  });

  test('check JavaScript for SSE code', async ({ page }) => {
    await page.goto('/');
    
    // Get the page source and check for SSE code
    const pageContent = await page.content();
    
    const hasConnectSSE = pageContent.includes('connectSSE');
    const hasEventSource = pageContent.includes('EventSource');
    const hasBroadcast = pageContent.includes('/events');
    
    console.log('Has connectSSE function:', hasConnectSSE);
    console.log('Has EventSource:', hasEventSource);
    console.log('Has /events endpoint reference:', hasBroadcast);
    
    if (!hasConnectSSE) {
      console.log('ERROR: SSE code is missing - server needs to be restarted with new code');
    }
  });

  test('monitor SSE connection in browser', async ({ page }) => {
    // Capture all console output
    const logs = [];
    page.on('console', msg => {
      logs.push(`[${msg.type()}] ${msg.text()}`);
    });
    
    // Capture network requests
    const requests = [];
    page.on('request', req => {
      requests.push({ url: req.url(), method: req.method() });
    });
    
    await page.goto('/');
    
    // Wait for potential SSE connection
    await page.waitForTimeout(3000);
    
    console.log('=== Console Logs ===');
    logs.forEach(log => console.log(log));
    
    console.log('=== Network Requests ===');
    const eventRequests = requests.filter(r => r.url.includes('/events'));
    console.log('SSE requests:', eventRequests);
    
    if (eventRequests.length === 0) {
      console.log('WARNING: No SSE connection was made');
    }
  });

});
