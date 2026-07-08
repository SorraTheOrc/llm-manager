/**
 * Model Display Extension (LP-0MR4ZIGDT004A3E1)
 *
 * Displays the resolved provider/model from proxy responses in Pi's status bar.
 *
 * Shows the format: `<selected-model> → <provider>/<model-id>`
 *
 * - Listens to `after_provider_response` to read the `X-Resolved-Model` header
 * - Listens to `model_select` to track the currently selected Pi model
 * - Updates the status bar on each request
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  // Track the currently selected model alias (e.g., "plan", "code")
  let selectedModel: string | null = null;

  // Track the resolved model from the last proxy response
  let resolvedModel: string | null = null;

  // Update the status display
  function updateStatus(ctx: { ui: { setStatus: (key: string, value: string) => void } }) {
    if (selectedModel && resolvedModel) {
      ctx.ui.setStatus("model-display", `${selectedModel} → ${resolvedModel}`);
    } else if (selectedModel) {
      ctx.ui.setStatus("model-display", `${selectedModel} → (pending)`);
    } else if (resolvedModel) {
      ctx.ui.setStatus("model-display", `→ ${resolvedModel}`);
    }
  }

  // Track model selection changes
  pi.on("model_select", (_event, ctx) => {
    // event.model has provider and id fields
    const model = _event.model;
    if (model && model.id) {
      selectedModel = model.id;
    }
    updateStatus(ctx);
  });

  // Extract resolved model from proxy response headers
  pi.on("after_provider_response", (event, ctx) => {
    // event.headers contains the normalized response headers
    // X-Resolved-Model header is lowercase in normalized form
    const headers = event.headers as Record<string, string>;
    const headerValue: string | undefined =
      headers["x-resolved-model"] ?? headers["X-Resolved-Model"];

    if (headerValue) {
      resolvedModel = headerValue;
    }
    updateStatus(ctx);
  });
}
