# QA Checklist — LP-0MN7LEGVX003YTSL — Improve log output

This document provides a brief manual QA checklist and example screenshots for the /logs UI improvements (severity badges, colors, and persistence).

Screenshots
- Desktop: `screenshots/logs-desktop.svg`
- Mobile: `screenshots/logs-mobile.svg`

Manual QA checklist

1. Load the Log page
   - Open: `http://localhost:8000/logs`
   - Verify the page loads and the two panes (Proxy log, Llama-server log) are visible.
   - Screenshot (Desktop): `screenshots/logs-desktop.svg`

2. Colourized badges
   - Confirm each log line shows a small badge on the left with one of: `E`, `W`, `I`, `D`.
   - Confirm the badge background colours correspond to severity (error=red, warn=orange, info=light, debug=muted).
   - Confirm the badge includes visible text label (not only colour).

3. Accessibility/Contrast
   - Toggle the colorize switch to ON and OFF and ensure badges are present or hidden accordingly.
   - Confirm badge text contrast against the badge background is sufficient (readable without relying on colour alone).
   - Use a contrast tool (e.g., browser devtools or an automated/a11y tool) to check contrast ratios for the badges.

4. Preference persistence
   - With the Colorize toggle set to OFF, reload the page and confirm the toggle remains OFF.
   - With the Colorize toggle set to ON, reload the page and confirm the toggle remains ON.

5. Initial log and live tail
   - Confirm the initial block (`--- initial log ---`) shows previous lines and the live tail appends new lines with badges preserved.

6. Mobile rendering
   - Open the Log page in a mobile viewport (or use the supplied mobile screenshot) and confirm badges, text wrapping, and toggle are usable.
   - Screenshot (Mobile): `screenshots/logs-mobile.svg`

Notes
- The screenshots included here are representative SVGs added to the PR to aid reviewers. Replace with real screenshots if preferred.
- Automated tests are included to validate the client/server severity parsing and CSS contrast expectations.
