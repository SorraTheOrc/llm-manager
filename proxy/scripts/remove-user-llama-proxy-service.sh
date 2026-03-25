#!/usr/bin/env bash
set -euo pipefail

# Remove a user systemd unit named 'llama-proxy' for a given user.
# Usage (as root): sudo ./remove-user-llama-proxy-service.sh <username>
# If run as non-root it will attempt to operate for the current user.

UNIT_NAME=llama-proxy

if [ "$#" -ge 1 ]; then
  TARGET_USER="$1"
else
  TARGET_USER="${SUDO_USER:-${USER:-}}"
fi

if [ -z "$TARGET_USER" ]; then
  echo "Usage: $0 <username>" >&2
  exit 2
fi

echo "[INFO] Target user: $TARGET_USER"

# Resolve home directory
HOME_DIR=$(getent passwd "$TARGET_USER" | cut -d: -f6)
if [ -z "$HOME_DIR" ]; then
  echo "Cannot determine home for user $TARGET_USER" >&2
  exit 3
fi

# If run as root, we will run systemctl --user commands as the target user.
AS_ROOT=0
if [ "$(id -u)" -eq 0 ]; then
  AS_ROOT=1
fi

echo "[STEP] Inspect current user unit status (best-effort)"
if [ "$AS_ROOT" -eq 1 ]; then
  sudo -u "$TARGET_USER" systemctl --user status "$UNIT_NAME" --no-pager || true
else
  systemctl --user status "$UNIT_NAME" --no-pager || true
fi

echo "[STEP] Stop and disable the user unit"
if [ "$AS_ROOT" -eq 1 ]; then
  sudo -u "$TARGET_USER" systemctl --user stop "$UNIT_NAME" 2>/dev/null || true
  sudo -u "$TARGET_USER" systemctl --user disable --now "$UNIT_NAME" 2>/dev/null || true
else
  systemctl --user stop "$UNIT_NAME" 2>/dev/null || true
  systemctl --user disable --now "$UNIT_NAME" 2>/dev/null || true
fi

# Backup and remove user unit files
USER_UNIT_PATH="$HOME_DIR/.config/systemd/user/${UNIT_NAME}.service"
BACKUP_DIR="/tmp"
TS=$(date +%s)
if [ -f "$USER_UNIT_PATH" ]; then
  BACKUP_PATH="${BACKUP_DIR}/${UNIT_NAME}.service.${TARGET_USER}.bak.${TS}"
  echo "[STEP] Backing up user unit file: $USER_UNIT_PATH -> $BACKUP_PATH"
  mv "$USER_UNIT_PATH" "$BACKUP_PATH"
else
  echo "[INFO] No user unit file at $USER_UNIT_PATH"
fi

USER_UNIT_PATH2="$HOME_DIR/.local/share/systemd/user/${UNIT_NAME}.service"
if [ -f "$USER_UNIT_PATH2" ]; then
  BACKUP_PATH2="${BACKUP_DIR}/${UNIT_NAME}.service.${TARGET_USER}.local.bak.${TS}"
  echo "[STEP] Backing up local user unit file: $USER_UNIT_PATH2 -> $BACKUP_PATH2"
  mv "$USER_UNIT_PATH2" "$BACKUP_PATH2"
fi

echo "[STEP] Reloading user systemd daemon (as user)"
if [ "$AS_ROOT" -eq 1 ]; then
  sudo -u "$TARGET_USER" systemctl --user daemon-reload || true
  sudo -u "$TARGET_USER" systemctl --user mask "$UNIT_NAME" 2>/dev/null || true
else
  systemctl --user daemon-reload || true
  systemctl --user mask "$UNIT_NAME" 2>/dev/null || true
fi

echo "[INFO] Final user unit status (best-effort):"
if [ "$AS_ROOT" -eq 1 ]; then
  sudo -u "$TARGET_USER" systemctl --user status "$UNIT_NAME" --no-pager || true
  sudo -u "$TARGET_USER" systemctl --user is-enabled "$UNIT_NAME" 2>/dev/null || echo "disabled or not-found"
else
  systemctl --user status "$UNIT_NAME" --no-pager || true
  systemctl --user is-enabled "$UNIT_NAME" 2>/dev/null || echo "disabled or not-found"
fi

echo "[NOTE] If linger is enabled for this user (loginctl enable-linger $TARGET_USER), user services may still run even when not logged in. To check: sudo loginctl user-status $TARGET_USER"

echo "[DONE]"

exit 0
