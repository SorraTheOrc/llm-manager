#!/usr/bin/env bash
set -euo pipefail

# Setup script to enable user linger, ensure newuidmap/newgidmap are installed and setuid,
# and restart the user llama-proxy service. Designed to be run as root (via sudo).

USERNAME="rgardler"

if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root (sudo)." >&2
  exit 2
fi

echo "[INFO] Enabling linger for user: $USERNAME"
loginctl enable-linger "$USERNAME"

echo "[INFO] Checking for newuidmap/newgidmap"
MISSING_PKGS=()
if [ ! -x /usr/bin/newuidmap ] || [ ! -x /usr/bin/newgidmap ]; then
  echo "  newuidmap/newgidmap not present or not executable"
  MISSING_PKGS+=(uidmap)
fi

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
  echo "The following helper(s) appear missing: ${MISSING_PKGS[*]}"
  echo "I can attempt to install a likely package for your distribution. Continue? [y/N]"
  read -r ans
  if [ "${ans,,}" != "y" ]; then
    echo "Skipping package install. Please install 'uidmap' (Debian/Ubuntu) or 'shadow-utils'/'shadow' (Fedora/Arch) as appropriate and re-run this script." >&2
  else
    # Detect package manager
    if command -v apt-get >/dev/null 2>&1; then
      echo "[INFO] Detected apt-get; installing uidmap"
      apt-get update
      apt-get install -y uidmap
    elif command -v dnf >/dev/null 2>&1; then
      echo "[INFO] Detected dnf; installing shadow-utils"
      dnf install -y shadow-utils
    elif command -v yum >/dev/null 2>&1; then
      echo "[INFO] Detected yum; installing shadow-utils"
      yum install -y shadow-utils
    elif command -v pacman >/dev/null 2>&1; then
      echo "[INFO] Detected pacman; installing shadow"
      pacman -Sy --noconfirm shadow
    elif command -v zypper >/dev/null 2>&1; then
      echo "[INFO] Detected zypper; installing shadow-utils"
      zypper --non-interactive install shadow-utils
    else
      echo "[WARN] Could not detect package manager. Please install uidmap/newuidmap manually." >&2
    fi
  fi
fi

echo "[INFO] Ensuring newuidmap/newgidmap are setuid root"
if [ -e /usr/bin/newuidmap ]; then
  chown root:root /usr/bin/newuidmap || true
  chmod u+s /usr/bin/newuidmap || true
  ls -l /usr/bin/newuidmap
fi
if [ -e /usr/bin/newgidmap ]; then
  chown root:root /usr/bin/newgidmap || true
  chmod u+s /usr/bin/newgidmap || true
  ls -l /usr/bin/newgidmap
fi

echo "[INFO] Checking /etc/subuid and /etc/subgid for user mappings"
grep -E "^${USERNAME}:" /etc/subuid || echo "MISSING_SUBUID"
grep -E "^${USERNAME}:" /etc/subgid || echo "MISSING_SUBGID"

if ! grep -E "^${USERNAME}:" /etc/subuid >/dev/null 2>&1 || ! grep -E "^${USERNAME}:" /etc/subgid >/dev/null 2>&1; then
  echo "One or both of /etc/subuid or /etc/subgid are missing an entry for $USERNAME."
  echo "Typical entry format: ${USERNAME}:100000:65536"
  echo "Would you like me to add a default mapping (${USERNAME}:100000:65536) to both files? This will back up the originals first. [y/N]"
  read -r ans2
  if [ "${ans2,,}" = "y" ]; then
    cp /etc/subuid /etc/subuid.bak.$(date +%s) || true
    cp /etc/subgid /etc/subgid.bak.$(date +%s) || true
    if ! grep -E "^${USERNAME}:" /etc/subuid >/dev/null 2>&1; then
      echo "${USERNAME}:100000:65536" >> /etc/subuid
      echo "Added to /etc/subuid"
    fi
    if ! grep -E "^${USERNAME}:" /etc/subgid >/dev/null 2>&1; then
      echo "${USERNAME}:100000:65536" >> /etc/subgid
      echo "Added to /etc/subgid"
    fi
  else
    echo "Skipping editing /etc/subuid and /etc/subgid. If these are missing, rootless containers will fail to create namespaces." >&2
  fi
fi

echo "[INFO] Reloading and restarting user service as $USERNAME"
# Use runuser or su/sudo to run the user systemctl commands as the target user
if command -v runuser >/dev/null 2>&1; then
  runuser -l "$USERNAME" -c 'systemctl --user daemon-reload && systemctl --user restart llama-proxy.service || true'
elif command -v su >/dev/null 2>&1; then
  su - "$USERNAME" -c 'systemctl --user daemon-reload && systemctl --user restart llama-proxy.service || true'
else
  echo "[WARN] Cannot run user systemctl command automatically; please run as the user: systemctl --user daemon-reload && systemctl --user restart llama-proxy.service" >&2
fi

echo "[INFO] Done. Check the user service logs with: sudo -u $USERNAME journalctl --user -u llama-proxy -n 200 --no-pager"
