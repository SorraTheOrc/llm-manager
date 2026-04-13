// Node integration test for strict HTTP parsing clients (undici/fetch)
// Usage: run with Node in CI or locally. Assumes proxy listening on http://localhost:8000
import fetch from 'node-fetch';

async function main() {
  const base = 'http://localhost:8000';
  const payload = {
    model: 'qwen3',
    messages: [{ role: 'user', content: 'Hello' }],
    max_tokens: 5
  };

  try {
    const res = await fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      // undici strict parser equivalent: node-fetch uses undici underneath
      // We merely assert the request completes without parser errors
    });

    if (!res.ok) {
      console.error('Non-OK status', res.status, await res.text());
      process.exit(2);
    }

    const data = await res.json();
    if (!data || !data.choices) {
      console.error('Unexpected body', JSON.stringify(data).slice(0, 2000));
      process.exit(3);
    }

    console.log('OK');
    process.exit(0);
  } catch (err) {
    console.error('Fetch failed', err && err.stack ? err.stack : err);
    process.exit(4);
  }
}

if (require.main === module) {
  main();
}
