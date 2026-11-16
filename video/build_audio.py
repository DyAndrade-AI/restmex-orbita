"""Genera una pista de voz por frase con `say` (Paulina, es_MX) y la línea de tiempo del video."""
import json, os, subprocess, wave, sys
from pathlib import Path
import numpy as np

OUT = Path(sys.argv[1])  # carpeta de trabajo (scratchpad)
HERE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent  # carpeta con script.json
RATE = os.environ.get("RATE", "176")  # palabras por minuto de `say`
OUT.mkdir(parents=True, exist_ok=True)
SR = 48000
LEAD, GAP, TAIL = 0.6, 0.35, 0.9  # silencio al inicio de escena, entre frases y al final

def tts(text, path):
    aiff = path.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Paulina", "-r", RATE, "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", str(SR), "-ac", "1", str(path)], check=True)
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

scenes, chunks, t = [], [], 0.0
for s, scene in enumerate(json.loads((HERE / "script.json").read_text())):
    start = t
    t += LEAD
    chunks.append(np.zeros(int(LEAD * SR), np.int16))
    phrases = []
    for p, text in enumerate(scene["phrases"]):
        pcm = tts(text, OUT / f"s{s:02d}p{p}.wav")
        d = len(pcm) / SR
        phrases.append({"text": text, "start": round(t, 3), "dur": round(d, 3)})
        chunks += [pcm, np.zeros(int(GAP * SR), np.int16)]
        t += d + GAP
    chunks.append(np.zeros(int(TAIL * SR), np.int16))
    t += TAIL
    scenes.append({"id": scene["id"], "start": round(start, 3), "end": round(t, 3), "phrases": phrases})

with wave.open(str(OUT / "voice.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(np.concatenate(chunks).tobytes())
assert abs(sum(len(c) for c in chunks) / SR - t) < 0.05, "la pista y la línea de tiempo no coinciden"
(OUT / "timeline.json").write_text(json.dumps({"total": round(t, 3), "scenes": scenes}, ensure_ascii=False, indent=1))
print(f"total {t:.1f}s, {len(scenes)} escenas")
