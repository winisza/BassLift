#!/bin/bash
# Buduje BassLift.app (Apple Silicon, podpis lokalny ad-hoc) i obraz dist/BassLift-<wersja>.dmg.
#   scripts/build_macos_app.sh
# Pierwszy raz kilka minut (pobiera Pythona dla aplikacji i instaluje zależności), potem szybciej.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_VENV=.venv-build
if [ ! -x "$BUILD_VENV/bin/briefcase" ]; then
  python3 -m venv "$BUILD_VENV"
  "$BUILD_VENV/bin/pip" install --disable-pip-version-check -q briefcase
fi
BRIEFCASE="$BUILD_VENV/bin/briefcase"

# Ikona .icns z logo/ (sips i iconutil są w macOS)
ICONSET=build/icons/basslift.iconset
mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s logo/basslift-icon-512.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  if [ $((s * 2)) -le 512 ]; then
    sips -z $((s * 2)) $((s * 2)) logo/basslift-icon-512.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  fi
done
iconutil -c icns "$ICONSET" -o build/icons/basslift.icns

# Briefcase instaluje wyłącznie gotowe koła (wheel). Pakiety wydane tylko jako źródła
# (np. midiutil) budujemy lokalnie do build/wheelhouse — pip w Briefcase ich tam szuka.
PIP="$BUILD_VENV/bin/pip"
PYV=$("$BUILD_VENV/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
WHEELHOUSE=build/wheelhouse
mkdir -p "$WHEELHOUSE"
for _ in $(seq 30); do
  if "$PIP" install --dry-run --disable-pip-version-check --ignore-installed --target build/.probe \
      --platform macosx_14_0_arm64 --python-version "$PYV" --only-binary :all: \
      --find-links "$WHEELHOUSE" -r requirements.txt > build/.probe.err 2>&1; then
    break
  fi
  # pip zgłasza brak koła na dwa sposoby: "satisfies the requirement X" albo listę
  # pod "no matching distributions available for your environment:"
  missing=$("$BUILD_VENV/bin/python" - build/.probe.err <<'PY'
import re, sys
err = open(sys.argv[1]).read()
found = re.findall(r"satisfies the requirement ([^ ;]+)", err)
block = re.search(r"no matching distributions available for your environment:\n((?:[ \t]+\S+[ \t]*\n)+)", err)
if block:
    found += block.group(1).split()
print(" ".join(dict.fromkeys(found)))
PY
)
  if [ -z "$missing" ]; then cat build/.probe.err; exit 1; fi
  for req in $missing; do
    echo "Buduję koło dla: $req"
    MACOSX_DEPLOYMENT_TARGET=14.0 "$PIP" wheel --quiet --disable-pip-version-check --no-deps \
      --wheel-dir "$WHEELHOUSE" "$req"
  done
done

APP_DIR=build/basslift/macos/app
if [ -d "$APP_DIR" ]; then
  "$BRIEFCASE" update macOS --update-requirements --update-resources --no-input
else
  "$BRIEFCASE" create macOS --no-input
fi

# .pyc z góry: działająca aplikacja nie może niczego zapisywać do swojego wnętrza (podpis)
"$BUILD_VENV/bin/python" -m compileall -q -j 0 "$APP_DIR/BassLift.app/Contents/Resources" || true

"$BRIEFCASE" build macOS --no-input
"$BRIEFCASE" package macOS --adhoc-sign --no-input
echo
echo "Gotowe: $APP_DIR/BassLift.app oraz $(ls -t dist/*.dmg | head -1)"
