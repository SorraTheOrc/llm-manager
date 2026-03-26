#!/usr/bin/env bash
set -euo pipefail

# Safe host-side cleanup script to remove/disable an installed systemd unit
# for llama-proxy. Designed to be run as root (via sudo).
#
# Usage: sudo ./remove-host-llama-proxy-service.sh
# This script will:
#  - stop and disable the unit if running
#  - back up the installed unit file to /tmp with a timestamp
#  - remove any wants/ symlink
#  - mask the unit
#  - reload systemd and reset failed state
#  - report status

UNIT_NAME="llama-proxy"
SYSTEM_UNIT_PATH="/etc/systemd/system/${UNIT_NAME}.service"
WANTS_LINK="/etc/systemd/system/multi-user.target.wants/${UNIT_NAME}.service"
BACKUP_DIR="/tmp"
TS=$(date +%s)

if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root (sudo)." >&2
  exit 2
fi

echo "[INFO] Cleaning up systemd unit: ${UNIT_NAME}"

echo "[STEP] Stopping and disabling unit (if active)"
systemctl stop "${UNIT_NAME}" 2>/dev/null || true
systemctl disable --now "${UNIT_NAME}" 2>/dev/null || true

if [ -f "${SYSTEM_UNIT_PATH}" ]; then
  BACKUP_PATH="${BACKUP_DIR}/${UNIT_NAME}.service.bak.${TS}"
  echo "[STEP] Backing up unit file: ${SYSTEM_UNIT_PATH} -> ${BACKUP_PATH}"
  mv "${SYSTEM_UNIT_PATH}" "${BACKUP_PATH}"
else
  echo "[INFO] No system unit file at ${SYSTEM_UNIT_PATH}"
fi

if [ -L "${WANTS_LINK}" ] || [ -f "${WANTS_LINK}" ]; then
  echo "[STEP] Removing wants-symlink: ${WANTS_LINK}"
  rm -f "${WANTS_LINK}"
else
  echo "[INFO] No wants symlink at ${WANTS_LINK}"
fi

echo "[STEP] Masking the unit to prevent accidental starts"
systemctl mask "${UNIT_NAME}" 2>/dev/null || true

echo "[STEP] Reloading systemd and resetting failed state"
systemctl daemon-reload
systemctl reset-failed || true

echo "[INFO] Final status for ${UNIT_NAME}:"
systemctl status "${UNIT_NAME}" --no-pager || true
echo
echo "[INFO] Unit enablement check:" 
systemctl is-enabled "${UNIT_NAME}" 2>/dev/null || echo "disabled or not-found"

echo
echo "[NOTE] If the unit was installed as a user unit, run the following as that user:" 
echo "  systemctl --user stop ${UNIT_NAME} && systemctl --user disable --now ${UNIT_NAME}"
echo "  mv ~/.config/systemd/user/${UNIT_NAME}.service /tmp/${UNIT_NAME}.service.bak.$TS"
echo "  systemctl --user daemon-reload && systemctl --user mask ${UNIT_NAME}"

echo "[DONE] Host-side cleanup completed. Backup (if any) stored under ${BACKUP_DIR}."

exit 0
