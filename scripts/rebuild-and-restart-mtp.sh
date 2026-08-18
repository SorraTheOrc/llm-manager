#!/usr/bin/env bash
# Rebuild llama.cpp with MTP support and restart the proxy stack.
# One-shot scheduled by cron for work item LP-0MSNI1B68001VE6C (Qwen MTP eval).
#
# LP-0MSXXKZOW0038XLK: on rebuild/deploy failure this script does NOT abort the
# restart. It restarts the stack (old binary — keeps service up), then re-attempts
# the deploy of the already-built artifact via `rebuild-llama.sh --deploy-only`
# (stop -> copy -> patchelf -> verify), then restarts once more so the new binary
# actually serves. Exit codes: 0 = stack healthy + deploy OK; 1 = stack healthy but
# deploy failed (old binary still serving); 2+ = stack never became healthy.
#
# Sequence:
#   1. scripts/rebuild-llama.sh (clone + cmake HIP + build + ordered deploy)
#   2. Verify deployed binary reports --spec-type draft-mtp
#   3. Restart the proxy stack (proxy/scripts/start-proxy.sh --restart; kills all
#      llama-server/uvicorn/TTS itself, incl. force-kill + port-free fallbacks)
#   4. If step 1 failed but the built artifact exists: re-attempt deploy
#      (--deploy-only) and restart the stack again
#   5. Poll /health until ready
#
# All output -> ${MTP_LOG} (/var/log/llama-proxy/rebuild-mtp.log)
#
# Env overrides (used by tests and operators):
#   MTP_LOG          log file             MTP_REPO_ROOT      repo root (default /home/rgardler/projects/llm)
#   MTP_REBUILD_CMD  rebuild command      MTP_NEW_BIN        deploy path to verify
#   MTP_ARTIFACT     built artifact path  MTP_SKIP_DATE_GATE skip the date gate (tests only)
#   MTP_TARGET_DATE  scheduled date       MTP_HEALTH_URL     health endpoint URL

set -uo pipefail

LOG="${MTP_LOG:-/var/log/llama-proxy/rebuild-mtp.log}"
REPO_ROOT="${MTP_REPO_ROOT:-/home/rgardler/projects/llm}"
NEW_BIN="${MTP_NEW_BIN:-/home/rgardler/llama.cpp/build/bin/llama-server}"
REBUILD_CMD="${MTP_REBUILD_CMD:-bash $REPO_ROOT/scripts/rebuild-llama.sh}"
ARTIFACT="${MTP_ARTIFACT:-/tmp/llama_rebuild/build/bin/llama-server}"
HEALTH_URL="${MTP_HEALTH_URL:-http://localhost:8000/health}"
TARGET_DATE="${MTP_TARGET_DATE:-2026-08-18}"

# One-shot: remove our cron entry on ANY fire (a later fire means the machine was
# off at 00:30 and the changes were not deployed — state re-verified by operator).
crontab -l 2>/dev/null | grep -v 'rebuild-and-restart-mtp.sh' | crontab -
echo "$(date '+%F %T %Z') rebuild-and-restart-mtp.sh: cron entry removed (one-shot fired)" >> "$LOG"

# Date gate: only rebuild on the scheduled night (unless skipped for tests).
if [[ -z "${MTP_SKIP_DATE_GATE:-}" && "$(date +%F)" != "$TARGET_DATE" ]]; then
    echo "$(date '+%F %T %Z') rebuild-and-restart-mtp.sh: not target date ($TARGET_DATE), exiting" >> "$LOG"
    exit 0
fi

teelog() { echo "$@" | tee -a "$LOG"; }

teelog "=== MTP Rebuild Started: $(date '+%F %T %Z') ==="
teelog "env PATH=$PATH"

cd "$REPO_ROOT" || { teelog "ERROR: cannot cd to $REPO_ROOT"; exit 1; }

# --- Step 1: rebuild llama.cpp from master (clone + HIP build + ordered deploy) ---
teelog "--- Step 1: rebuild-llama.sh ---"
FULL_REBUILD="$REBUILD_CMD --json --output /tmp/mtp-rebuild-status.json"
bash -c "$FULL_REBUILD" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
if [[ $RC -ne 0 ]]; then
    teelog "WARNING: rebuild-llama.sh failed (rc=$RC); NOT aborting — restarting stack first, deploy will be re-attempted"
fi

# --- Step 2: verify the deployed binary has MTP support (best effort) ---
teelog "--- Step 2: verify MTP support in $NEW_BIN ---"
if "$NEW_BIN" --help 2>&1 | grep -q "draft-mtp"; then
    teelog "VERIFIED: binary supports --spec-type draft-mtp"
else
    teelog "WARNING: binary does NOT list draft-mtp in --help; continuing anyway"
fi
"$NEW_BIN" --version 2>&1 | head -2 | tee -a "$LOG"

restart_stack() {
    cd "$REPO_ROOT/proxy" || { teelog "ERROR: cannot cd to $REPO_ROOT/proxy"; return 1; }
    nohup bash scripts/start-proxy.sh --restart &>/tmp/mtp-proxy-restart.log &
    teelog "start-proxy.sh --restart launched (pid $!), log: /tmp/mtp-proxy-restart.log"
    return 0
}

poll_health() {
    for i in $(seq 1 300); do
        if curl -s --max-time 5 "$HEALTH_URL" 2>/dev/null \
            | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('ready') else 1)" 2>/dev/null; then
            teelog "Proxy healthy after ${i}s!"
            return 0
        fi
        sleep 1
    done
    teelog "ERROR: proxy not healthy within 300s; inspect /tmp/mtp-proxy-restart.log"
    return 1
}

# --- Step 3: restart the proxy stack (kill old servers first — cures ETXTBSY) ---
teelog "--- Step 3: restart proxy stack ---"
restart_stack || { teelog "ERROR: restart_stack failed"; exit 2; }

# --- Step 4: if the build succeeded but the deploy did not, re-attempt the deploy
# of the existing artifact now that no process holds the old binary ---
DEP_OK=0
if [[ $RC -ne 0 && -f "$ARTIFACT" ]]; then
    teelog "--- Step 4: re-attempt deploy of existing artifact ($ARTIFACT) ---"
    DEP_CMD="$REBUILD_CMD --deploy-only --json"
    bash -c "$DEP_CMD" 2>&1 | tee -a "$LOG"
    DEP_RC=${PIPESTATUS[0]}
    if [[ $DEP_RC -ne 0 ]]; then
        teelog "WARNING: re-deploy failed (rc=$DEP_RC); old binary may still be in place"
    else
        DEP_OK=1
        teelog "Re-deploy OK; restarting stack to serve the new binary"
        restart_stack || { teelog "ERROR: second restart_stack failed"; exit 2; }
        sleep "${MTP_POST_REDEPLOY_SLEEP:-5}"
    fi
elif [[ $RC -ne 0 ]]; then
    teelog "WARNING: no artifact at $ARTIFACT; skip re-deploy (build never completed)"
fi

# --- Step 5: poll /health until ready ---
teelog "--- Step 5: waiting for proxy health ---"
if poll_health; then
    if [[ $RC -eq 0 || $DEP_OK -eq 1 ]]; then
        teelog "=== MTP Rebuild Complete: $(date '+%F %T %Z') ==="
        exit 0
    fi
    teelog "=== MTP Rebuild Finished with deploy failure (stack healthy, old binary serving): $(date '+%F %T %Z') ==="
    exit 1
fi

teelog "ERROR: proxy not healthy within 300s; inspect /tmp/mtp-proxy-restart.log"
exit 2