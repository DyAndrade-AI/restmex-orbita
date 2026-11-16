"""Sintetiza la música de fondo y la mezcla con la voz (ducking). Salida: mix.wav estéreo 48 kHz."""
import json, sys, wave
from pathlib import Path
import numpy as np

work = Path(sys.argv[1])
SR = 48000
tl = json.loads((work / "timeline.json").read_text())
with wave.open(str(work / "voice.wav")) as w:
    voice = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
N = len(voice) + SR  # 1 s de cola
t = np.arange(N) / SR
rng = np.random.default_rng(7)
hz = lambda m: 440 * 2 ** ((m - 69) / 12)

# progresión La m – Fa – Do – Sol, 8 s por acorde
CHORDS = [[57, 60, 64, 69], [53, 57, 60, 65], [48, 52, 55, 60], [55, 59, 62, 67]]
BAR = 8.0
L, R = np.zeros(N), np.zeros(N)
nbars = int(np.ceil(N / SR / BAR)) + 1
for b in range(nbars):
    notes = CHORDS[b % 4]
    s0, s1 = int((b * BAR - 1.5) * SR), int(((b + 1) * BAR + 1.5) * SR)
    s0c, s1c = max(0, s0), min(N, s1)
    if s0c >= s1c:
        continue
    tt = t[s0c:s1c]
    u = (np.arange(s0c, s1c) - s0) / (s1 - s0)
    env = np.sin(np.pi * u) ** 1.5  # fundido entre acordes
    for m in notes:
        f = hz(m)
        for det, side in ((0.9985, L), (1.0015, R)):
            side[s0c:s1c] += env * (np.sin(2 * np.pi * f * det * tt) + 0.18 * np.sin(4 * np.pi * f * det * tt)) * 0.05
    root = hz(notes[0] - 12)
    bass = env * np.sin(2 * np.pi * root * tt) * 0.09
    L[s0c:s1c] += bass; R[s0c:s1c] += bass

# arpegio de campanas, corcheas a 0.5 s
arp = np.zeros(N)
pattern = [0, 2, 1, 3, 2, 1, 3, 2]
for k in range(int(N / SR / 0.5)):
    st = k * 0.5
    notes = CHORDS[int(st // BAR) % 4]
    f = hz(notes[pattern[k % 8]] + 12)
    a, z = int(st * SR), min(N, int((st + 1.6) * SR))
    tt = (np.arange(a, z) - a) / SR
    arp[a:z] += (np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(6 * np.pi * f * tt)) * np.exp(-tt * 3.2) * np.minimum(1, tt * 200) * 0.035
for d, g, side in ((0.375, 0.35, R), (0.75, 0.2, L)):  # eco estéreo
    sh = int(d * SR); e = np.zeros(N); e[sh:] = arp[:-sh] * g; side += e
L += arp; R += arp

# transiciones: soplo de ruido en cada cambio de escena + golpe grave en el título y el cierre
def whoosh(center, gain):
    a, z = int((center - 0.6) * SR), int((center + 0.5) * SR)
    a, z = max(0, a), min(N, z)
    n = rng.standard_normal(z - a)
    k = 60
    n = np.convolve(n, np.ones(k) / k, mode="same")  # paso bajo
    u = np.linspace(0, 1, z - a)
    env = np.where(u < 0.55, (u / 0.55) ** 2, ((1 - u) / 0.45) ** 1.5)
    L[a:z] += n * env * gain; R[a:z] += n * env * gain * 0.85

def boom(at, gain):
    a, z = int(at * SR), min(N, int((at + 2.5) * SR))
    tt = (np.arange(a, z) - a) / SR
    f = 40 + 50 * np.exp(-tt * 4)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 1.8) * gain
    L[a:z] += s; R[a:z] += s

for sc in tl["scenes"][1:]:
    whoosh(sc["start"], 0.5)
scenes = {s["id"]: s for s in tl["scenes"]}
title = scenes["marca"]["phrases"][0] if "marca" in scenes else scenes["intro"]["phrases"][2]  # aparición del título
boom(title["start"] - 0.2, 0.5)
boom(scenes["cierre"]["phrases"][0]["start"] - 0.2, 0.5)

# ducking: la música baja cuando habla la voz
v = np.zeros(N); v[:len(voice)] = voice
win = int(0.05 * SR)
rms = np.sqrt(np.convolve(v ** 2, np.ones(win) / win, mode="same"))
active = (rms > 0.01).astype(float)
duck = np.empty(N); g = 1.0
att, rel = np.exp(-1 / (0.08 * SR)), np.exp(-1 / (0.6 * SR))
target = 1 - 0.55 * active
for i in range(0, N, 64):  # suavizado por bloques
    tg = target[i]
    c = att if tg < g else rel
    g = tg + (g - tg) * c ** 64
    duck[i:i + 64] = g
fade = np.minimum(1, t / 2.0) * np.minimum(1, (t[-1] - t) / 4.0)
music_gain = 0.55 * duck * fade
vpk = np.abs(v).max()
vg = 0.8 / vpk
outL = v * vg + L * music_gain
outR = v * vg + R * music_gain
pk = max(np.abs(outL).max(), np.abs(outR).max())
if pk > 0.98:
    outL, outR = outL * 0.98 / pk, outR * 0.98 / pk
st = np.stack([outL, outR], 1)
with wave.open(str(work / "mix.wav"), "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((st * 32767).astype(np.int16).tobytes())
print(f"mix.wav {N / SR:.1f}s, pico {pk:.2f}")
