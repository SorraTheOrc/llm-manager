#!/usr/bin/env bash
# Gather diagnostics to determine what starts the llama-proxy process on reboot.
# Safe, read-only. Run as root for full info: sudo bash diagnose-llama-proxy-startup.sh

set -u

TS=$(date --rfc-3339=seconds 2>/dev/null || date)
echo "== diagnose-llama-proxy-startup.sh - $TS =="

echo
echo "[1] Processes matching known patterns (uvicorn, llama-proxy, start-llama, distrobox, llama-server)"
ps -eo pid,ppid,user,uid,cmd --no-headers | egrep -i 'uvicorn|llama-proxy|start-llama|distrobox|llama-server' || true

echo
echo "[2] For each candidate PID show cgroup, exe and process line"
PIDS=$(ps -eo pid,cmd --no-headers | egrep -i 'uvicorn|llama-proxy|start-llama|distrobox|llama-server' | awk '{print $1}' | tr '\n' ' ')
if [ -z "$PIDS" ]; then
  echo "No matching PIDs found."
else
  for PID in $PIDS; do
    echo
    echo "---- PID: $PID ----"
    echo "cgroup:"; cat /proc/$PID/cgroup 2>/dev/null || echo "  (no /proc/$PID/cgroup)"
    echo "exe:"; readlink -f /proc/$PID/exe 2>/dev/null || echo "  (no /proc/$PID/exe)"
    echo "ps:"; ps -o pid,ppid,user,cmd -p $PID --no-headers || true

    # Try to infer owning unit from cgroup
    CG=$(cat /proc/$PID/cgroup 2>/dev/null | awk -F: '{print $3}' | tr '\n' ' ')
    if echo "$CG" | egrep -q 'system.slice|user.slice|/'; then
      echo "inferred cgroup: $CG"
      # If user-<uid> present, extract uid
      if echo "$CG" | egrep -q 'user-([0-9]+)\.slice'; then
        UID_NUM=$(echo "$CG" | sed -n 's/.*user-\([0-9]\+\)\.slice.*/\1/p')
        if [ -n "$UID_NUM" ]; then
          USERNAME=$(getent passwd "$UID_NUM" | cut -d: -f1 || true)
          echo "possible user: uid=$UID_NUM username=${USERNAME:-<unknown>}"
        fi
      fi
    fi
  done
fi

echo
echo "[3] systemd units (active/all) mentioning 'llama' or 'proxy'"
systemctl list-units --type=service --all | egrep -i 'llama|proxy' || true
echo
systemctl list-unit-files | egrep -i 'llama|proxy' || true

echo
echo "[4] If a user unit may be involved, and you're root, show user unit status for detected users"
# Find user ids that may have units from pids cgroups
if [ -n "$PIDS" ]; then
  UIDS=$(for PID in $PIDS; do cat /proc/$PID/cgroup 2>/dev/null | sed -n 's/.*user-\([0-9]\+\)\.slice.*/\1/p' || true; done | sort -u | tr '\n' ' ')
  for UID in $UIDS; do
    if [ -n "$UID" ]; then
      USERNAME=$(getent passwd "$UID" | cut -d: -f1 || true)
      echo
      echo "-- user candidate: uid=$UID username=${USERNAME:-<unknown>} --"
      if [ "$(id -u)" -eq 0 ]; then
        echo "systemctl --user status (as $USERNAME):"
        sudo -u "${USERNAME}" systemctl --user status llama-proxy --no-pager || true
        echo "loginctl user-status $USERNAME:"; loginctl user-status "$USERNAME" || true
      else
        echo "Run as root to inspect user units for $USERNAME: sudo -u $USERNAME systemctl --user status llama-proxy"
      fi
    fi
  done
fi

echo
echo "[5] Additional checks"
echo "- Is there an installed unit file under /etc/systemd/system or /lib/systemd/system or /usr/lib/systemd/system?"
echo "  Candidates:"; ls -l /etc/systemd/system/*llama* 2>/dev/null || true; ls -l /etc/systemd/system/*proxy* 2>/dev/null || true
echo "  /lib/systemd/system:"; ls -l /lib/systemd/system/*llama* 2>/dev/null || true; ls -l /lib/systemd/system/*proxy* 2>/dev/null || true
echo "  /usr/lib/systemd/system:"; ls -l /usr/lib/systemd/system/*llama* 2>/dev/null || true; ls -l /usr/lib/systemd/system/*proxy* 2>/dev/null || true

echo
echo "[6] Helpful next commands (not run by this script):"
echo "  sudo ps -ef | egrep 'uvicorn|llama-proxy|start-llama|distrobox'"
echo "  sudo systemctl status <unit-name>"
echo "  sudo systemctl disable --now <unit-name> && sudo systemctl mask <unit-name>"

echo
echo "== end =="
