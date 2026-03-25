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

  test('check SSE endpoint directly', async ({ request }) => {
    // Test the /events endpoint directly
    const response = await request.get('/events', {
      headers: {
        'Accept': 'text/event-stream'
      }
    });
    
    console.log('SSE endpoint status:', response.status());
    console.log('SSE content-type:', response.headers()['content-type']);
    
    // We can't easily read SSE with request API, but we can check it exists
    expect(response.status()).toBe(200);
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
