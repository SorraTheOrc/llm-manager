# Oversized-session correlation (2026-08-26)

Proxy-log timeline (timestamped) correlated with llama-server prefill/decode evidence (no timestamps in llama.cpp logs — attributed by file rotation window). Session context = `estimated_tokens` at routing time.

## Totals

- Routing checks: **4624** | sessions: **180**
- context_pressure warnings: **1731** | routing_skip_local: **1918**
- Estimated prefill work: **367,721,904 tokens**
- **Wasted prefill (ratio > 1.0): 281,163,994 tokens (76.5% of all prefill)**

## Hourly timeline

| Hour | Pressure | Skips | Dispatch denied | Upstream 5xx | Routing checks |
|---|---|---|---|---|---|
| 00:00 | 75 | 56 | 15 | 0 | 168 |
| 01:00 | 0 | 0 | 32 | 0 | 188 |
| 02:00 | 134 | 142 | 32 | 0 | 298 |
| 03:00 | 59 | 78 | 22 | 0 | 230 |
| 04:00 | 0 | 22 | 4 | 0 | 169 |
| 05:00 | 49 | 94 | 0 | 0 | 187 |
| 06:00 | 48 | 81 | 9 | 0 | 197 |
| 07:00 | 33 | 35 | 0 | 0 | 83 |
| 08:00 | 25 | 29 | 1 | 0 | 136 |
| 09:00 | 20 | 20 | 6 | 0 | 89 |
| 10:00 | 104 | 131 | 47 | 0 | 266 |
| 11:00 | 78 | 88 | 21 | 0 | 246 |
| 12:00 | 73 | 103 | 0 | 0 | 146 |
| 13:00 | 111 | 124 | 23 | 0 | 248 |
| 14:00 | 81 | 65 | 12 | 0 | 234 |
| 15:00 | 29 | 27 | 4 | 0 | 110 |
| 16:00 | 18 | 18 | 16 | 0 | 177 |
| 17:00 | 33 | 33 | 0 | 0 | 144 |
| 18:00 | 213 | 209 | 22 | 0 | 423 |
| 19:00 | 79 | 88 | 7 | 0 | 177 |
| 20:00 | 281 | 268 | 0 | 0 | 337 |
| 21:00 | 183 | 202 | 33 | 0 | 366 |
| 22:00 | 5 | 5 | 0 | 0 | 5 |

## Top sessions by wasted prefill work

| Session | Mode | Checks | Peak est. ctx | Prefill (tokens) | Wasted (tokens) | Checks ratio>1 | Skips | Pressure |
|---|---|---|---|---|---|---|---|---|
| `herdr-1787705716-3748616-11252` | fast | 387 | 651,408 | 98,611,313 | 94,577,628 | 317 | 355 | 314 |
| `herdr-1787710054-3791143-829` | fast | 258 | 421,242 | 42,568,185 | 40,672,033 | 213 | 240 | 213 |
| `herdr-1787766452-1454318-26884` | fast | 218 | 221,764 | 28,747,962 | 27,272,182 | 182 | 182 | 182 |
| `herdr-1787740130-712957-12873` | fast | 215 | 240,347 | 25,621,383 | 24,395,782 | 183 | 183 | 189 |
| `herdr-1787706917-3753330-22105` | fast | 217 | 162,389 | 24,453,426 | 23,271,542 | 194 | 199 | 157 |
| `herdr-1787768817-1449219-17291` | fast | 84 | 188,047 | 13,651,588 | 13,379,012 | 79 | 82 | 82 |
| `herdr-1787733596-13203-18690` | fast | 131 | 209,485 | 14,217,951 | 12,298,457 | 71 | 71 | 71 |
| `herdr-1787708398-1112069-13865` | fast | 114 | 134,021 | 10,644,078 | 9,485,956 | 89 | 89 | 96 |
| `herdr-1787727402-4185060-12919` | fast | 111 | 127,298 | 9,377,295 | 6,842,690 | 64 | 80 | 74 |
| `herdr-1787707304-1104300-6932` | fast | 91 | 130,393 | 6,891,430 | 5,962,672 | 58 | 58 | 58 |
| `herdr-1787710532-3792736-8267` | fast | 99 | 121,247 | 6,648,260 | 4,976,605 | 50 | 58 | 22 |
| `herdr-1787705413-3747206-22481` | fast | 50 | 167,364 | 5,258,408 | 4,920,560 | 37 | 38 | 37 |
| `herdr-1787691466-2847149-3752` | fast | 33 | 164,897 | 4,685,489 | 4,620,105 | 32 | 32 | 32 |
| `herdr-1787702928-1073080-5826` | fast | 148 | 109,169 | 9,149,009 | 4,138,747 | 50 | 52 | 52 |
| `herdr-1787776862-1503284-4433` | fast | 55 | 108,601 | 4,271,002 | 1,946,063 | 20 | 55 | 37 |

## llama-server decode/prefill evidence (window)

- Files in window: llama-server.14.log, llama-server.13.log, llama-server.12.log, llama-server.11.log, llama-server.10.log, llama-server.9.log, llama-server.8.log, llama-server.7.log
- Decode observations: **2532** | median 22.89 t/s | min 0.18 t/s
- **Slow decodes (< 1 t/s): 33**
- Prefill events: 2532 | total 5,179,736 tokens | max prefill 63,256 tokens
- Slowest examples: 0.18 t/s (58 tok), 0.21 t/s (93 tok), 0.22 t/s (59 tok), 0.24 t/s (124 tok), 0.25 t/s (114 tok)

## Caveats and methodology

- **llama.cpp logs carry no timestamps**; decode/prefill evidence is attributed by rotation-file close time, not hour-exact event time.
- **Proxy log gap on Aug 26 22:00-24:00**: rotated logs stop at 22:00:22 (proxy.log.2026-08-26_16) and resume at 00:00:03 Aug 27 (proxy.log.2026-08-27_01); ~2h of event data (including most of hour 22) is absent. Earlier cited figures for hour 22 (4,541 fallbacks, 280 backend 5xx, 42.7M prefill tokens, max 85,724) predate the calendar-day reconstruction and cover other windows; this report recomputes from the calendar day with the gap called out.
- **Wasted prefill** counts ``estimated_tokens`` on routing checks whose context exceeded the per-slot clamp (ratio > 1.0): such a session can never be resident in one slot, so every turn is a full re-prefill that persists no reusable KV — the prefill work is lost.

