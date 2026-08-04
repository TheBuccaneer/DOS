#!/usr/bin/env bash
# Explicit real-run wrapper: runs the full Llama campaign and powers off afterwards.
set -Eeuo pipefail

CONTROL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/llama_server_waiting_control.sh"
KEEPALIVE_PID=""

fail(){ echo "FEHLER: $*" >&2; exit 1; }

if test "${1:-}" != "--run-and-shutdown"; then
  cat <<USAGE
Dieses Skript startet eine REALE 32-Episoden-Kampagne und fährt den Rechner
anschließend auch bei Fehler oder Ctrl+C herunter.

Expliziter Aufruf:
  bash "$0" --run-and-shutdown

Ohne diese Option startet nichts.
USAGE
  exit 2
fi

test -x "$CONTROL" || fail "Control-Skript fehlt: $CONTROL"
command -v sudo >/dev/null 2>&1 || fail "sudo fehlt"
command -v shutdown >/dev/null 2>&1 || command -v systemctl >/dev/null 2>&1 || fail "Kein Shutdown-Kommando gefunden"

echo "=== SUDO-AUTHENTIFIZIERUNG ==="
sudo -v || fail "sudo-Authentifizierung fehlgeschlagen"
(
  while sleep 45; do sudo -n -v || exit 1; done
) &
KEEPALIVE_PID=$!

poweroff_on_exit(){
  local rc=$?
  trap - EXIT HUP INT TERM
  if test -n "${KEEPALIVE_PID:-}"; then
    kill "$KEEPALIVE_PID" 2>/dev/null || true
    wait "$KEEPALIVE_PID" 2>/dev/null || true
  fi
  echo
  echo "=== LLAMA-KAMPAGNE BEENDET ==="
  echo "RUN_RC=$rc"
  echo "RECHNER_WIRD_JETZT_HERUNTERGEFAHREN"
  sync || true
  sudo -n shutdown -h now || sudo -n systemctl poweroff || echo "FEHLER: Herunterfahren fehlgeschlagen." >&2
  exit "$rc"
}
trap poweroff_on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

bash "$CONTROL" --campaign
