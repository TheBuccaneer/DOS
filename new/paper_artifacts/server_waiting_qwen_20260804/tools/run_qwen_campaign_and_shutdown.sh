#!/usr/bin/env bash
# Starts the real Qwen campaign in the console and powers the PC off afterwards,
# regardless of whether the campaign succeeds or fails.
set -Eeuo pipefail

BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_SCRIPT="$BASE_DIR/run_qwen_campaign_headless_fixed.sh"
SUDO_KEEPALIVE_PID=""

fail() {
    echo "FEHLER: $*" >&2
    exit 1
}

test -x "$CAMPAIGN_SCRIPT" || fail "Kampagnenskript fehlt oder ist nicht ausführbar: $CAMPAIGN_SCRIPT"
command -v sudo >/dev/null 2>&1 || fail "sudo fehlt"
command -v shutdown >/dev/null 2>&1 || command -v systemctl >/dev/null 2>&1 || fail "Kein Shutdown-Kommando gefunden"

echo "=== SUDO-AUTHENTIFIZIERUNG ==="
echo "Bitte jetzt einmal das sudo-Passwort eingeben."
sudo -v || fail "sudo-Authentifizierung fehlgeschlagen"

# Keep the sudo timestamp alive while the multi-hour campaign runs.
(
    while sleep 45; do
        sudo -n -v || exit 1
    done
) &
SUDO_KEEPALIVE_PID=$!

shutdown_on_exit() {
    local rc=$?
    trap - EXIT HUP INT TERM

    if test -n "${SUDO_KEEPALIVE_PID:-}"; then
        kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
        wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    fi

    echo
    echo "=== KAMPAGNE BEENDET ==="
    echo "RUN_RC=$rc"
    echo "RECHNER_WIRD_JETZT_HERUNTERGEFAHREN"
    sync || true

    if ! sudo -n shutdown -h now; then
        sudo -n systemctl poweroff || echo "FEHLER: Automatisches Herunterfahren fehlgeschlagen." >&2
    fi

    exit "$rc"
}

# From here on, success, runner failure, Ctrl+C, or termination all lead to shutdown.
trap shutdown_on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

"$CAMPAIGN_SCRIPT" --run
