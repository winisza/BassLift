"""Test dymny okna aplikacji — pełna ekstrakcja przez prawdziwe UI, zapis plików i schowek.

Działa także w zbudowanej aplikacji (bez okien dialogowych, schowek przywracany):
    BASSLIFT_SMOKETEST=/tmp/wynik.json BassLift.app/Contents/MacOS/BassLift
albo z repo: python scripts/desktop_smoketest.py wynik.json
"""
import json
import os
import subprocess
import time
from pathlib import Path

# 10 s linii basu (E1/A1/D2, ósemki, 120 BPM) + stopa na ćwierćnutach — WAV generowany w stronie
MAKE_WAV = r"""
(() => {
  const sr = 44100, n = sr * 10, pcm = new Int16Array(n);
  const pitches = [28, 28, 33, 33, 38, 38, 33, 28];
  for (let k = 0; k < 40; k++) {
    const f = 440 * Math.pow(2, (pitches[k % 8] - 69) / 12), t0 = Math.floor((0.25 + k * 0.25) * sr);
    for (let i = 0; i < sr * 0.22 && t0 + i < n; i++) {
      const t = i / sr, env = Math.exp(-t * 6);
      pcm[t0 + i] += 9000 * env * (Math.sin(2 * Math.PI * f * t) + 0.5 * Math.sin(4 * Math.PI * f * t));
    }
    if (k % 2 === 0) for (let i = 0; i < sr * 0.1 && t0 + i < n; i++) {
      const t = i / sr; pcm[t0 + i] += 7000 * Math.exp(-t * 30) * Math.sin(2 * Math.PI * (50 + 60 * Math.exp(-t * 40)) * t);
    }
  }
  const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8, 'WAVEfmt '); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, n * 2, true);
  new Int16Array(buf, 44).set(pcm);
  const dt = new DataTransfer(); dt.items.add(new File([buf], 'smoketest.wav', { type: 'audio/wav' }));
  const inp = document.getElementById('fileInput'); inp.files = dt.files; inp.dispatchEvent(new Event('change'));
  document.getElementById('transcriptionEngine').value = 'crepe';
  document.getElementById('exportMidi').checked = true;
  document.getElementById('exportBass').checked = true;
  document.getElementById('runBtn').click();
  return 'started';
})()
"""


def run(window, out: Path, save_dir: Path):
    """Klika po UI w oknie aplikacji i zapisuje raport JSON; na koniec zamyka okno."""
    js = window.evaluate_js
    report, t0 = {}, time.time()
    try:
        while js("document.readyState") != "complete" or not js("!!(window.pywebview && window.pywebview.api)"):
            time.sleep(0.2)
        time.sleep(1.0)  # checkBackend
        # trwałość ustawień między uruchomieniami: wartość zapisana przez poprzedni test
        report["pref_from_previous_run"] = js("localStorage.getItem('basslift_smoketest')")
        js(f"localStorage.setItem('basslift_smoketest', '{int(time.time())}')")
        report["backend_hidden"] = js("getComputedStyle(document.getElementById('backendSection')).display === 'none'")
        report["fonts_ok"] = js("document.fonts.check('16px \"DM Sans\"') && document.fonts.check('16px \"Space Mono\"')")
        report["start"] = js(MAKE_WAV)
        stages = []
        while time.time() - t0 < 240:
            label = js("(document.querySelector('#stepList .step.active span') || {}).textContent || ''")
            if label and (not stages or stages[-1] != label.split(' · ')[0]):
                stages.append(label.split(' · ')[0])
            if js("document.getElementById('tabOutput').classList.contains('show')"):
                break
            time.sleep(0.5)
        report["seconds"] = round(time.time() - t0, 1)
        report["stages"] = stages
        report["error"] = js("document.getElementById('errorBox').textContent")
        report["pills"] = js("[...document.querySelectorAll('#tabMeta .meta-pill')].map(p => p.textContent)")
        tab = js("tabData") or ""
        report["tab_first_lines"] = tab.splitlines()[:6]

        # z Findera aplikacja startuje bez LANG — pbpaste oddałby wtedy MacRoman zamiast UTF-8
        env = {**os.environ, "LANG": "en_US.UTF-8"}
        before = subprocess.run(["pbpaste"], capture_output=True, env=env).stdout
        for btn in ("downloadTabBtn", "downloadMidiBtn", "downloadMusicXmlBtn", "downloadBassBtn", "copyBtn"):
            js(f"document.getElementById('{btn}').click()")
            time.sleep(1.0)
        report["clipboard_has_tab"] = subprocess.run(["pbpaste"], capture_output=True, env=env).stdout == tab.encode()
        subprocess.run(["pbcopy"], input=before, env=env)  # przywróć schowek (bajt w bajt)
        report["saved"] = {p.name: p.stat().st_size for p in save_dir.iterdir()}
    except Exception as e:  # raport zamiast wiszącego okna
        report["exception"] = repr(e)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    window.destroy()
