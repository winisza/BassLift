#!/bin/bash
# BassLift — uruchom dwuklikiem w Finderze (macOS). Na Linuksie: ./BassLift.command
cd "$(dirname "$0")" || exit 1

PY=""
for cand in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1 &&
     "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$cand"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "BassLift potrzebuje Pythona 3.10 lub nowszego."
  echo "Pobierz go z https://www.python.org/downloads/ i uruchom BassLift ponownie."
  [ "$(uname)" = "Darwin" ] && open "https://www.python.org/downloads/macos/"
  read -r -p "Naciśnij Enter, aby zamknąć okno." _
  exit 1
fi

"$PY" run.py "$@"
status=$?
if [ $status -ne 0 ]; then
  echo
  read -r -p "BassLift zakończył się błędem (kod $status). Naciśnij Enter, aby zamknąć okno." _
fi
exit $status
