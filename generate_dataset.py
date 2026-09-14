import os
import numpy as np
from scipy.io import wavfile

os.makedirs("dataset/healthy", exist_ok=True)
os.makedirs("dataset/infested", exist_ok=True)

sr = 22050
duration = 2.0  # seconds
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

print("Generating synthetic audio dataset...")

# Generate 50 Healthy Trunk Audios (Pink/ambient low-frequency noise)
for i in range(50):
    noise = np.random.normal(0, 0.05, len(t))
    ambient = 0.02 * np.sin(2 * np.pi * 60 * t) + noise
    wavfile.write(f"dataset/healthy/healthy_{i:02d}.wav", sr, (ambient * 32767).astype(np.int16))

# Generate 20 Infested Audios (Ambient noise + Larval chewing pulses)
for i in range(20):
    noise = np.random.normal(0, 0.05, len(t))
    signal = 0.02 * np.sin(2 * np.pi * 60 * t) + noise
    # Add chewing/gnawing transients (bursts of 1.5 kHz - 3 kHz sound)
    for _ in range(np.random.randint(4, 10)):
        start = np.random.randint(0, len(t) - 2000)
        click = np.sin(2 * np.pi * 2200 * t[:2000]) * np.hanning(2000) * 0.4
        signal[start:start+2000] += click
    wavfile.write(f"dataset/infested/infested_{i:02d}.wav", sr, (signal * 32767).astype(np.int16))

print("Dataset ready: 50 healthy samples, 20 infested anomaly samples.")