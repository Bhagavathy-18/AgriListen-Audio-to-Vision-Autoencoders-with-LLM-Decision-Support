import glob
import torch
import torch.nn as nn
import numpy as np
import librosa
from model import ConvAutoencoder

def extract_spectrogram(path):
    y, sr = librosa.load(path, sr=22050, duration=2.0)
    if len(y) < 22050 * 2:
        y = np.pad(y, (0, int(22050 * 2) - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    
    # NEW: Force exactly 88 frames by padding if necessary
    if norm.shape[1] < 88:
        norm = np.pad(norm, ((0, 0), (0, 88 - norm.shape[1])), mode='constant')
        
    return norm[:128, :88]

healthy_files = glob.glob("dataset/healthy/*.wav")
data = [extract_spectrogram(f) for f in healthy_files]
X_train = torch.tensor(np.array(data), dtype=torch.float32).unsqueeze(1)

model = ConvAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

print("Training Convolutional Autoencoder on Healthy Baseline...")
for epoch in range(25):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, X_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/25], Reconstruction Loss: {loss.item():.5f}")

# Calculate anomaly detection threshold
model.eval()
with torch.no_grad():
    recon = model(X_train)
    losses = torch.mean((recon - X_train) ** 2, dim=[1, 2, 3]).numpy()
    threshold = float(np.mean(losses) + 3 * np.std(losses))

print(f"\nTraining Complete. Baseline Anomaly Threshold: {threshold:.5f}")
torch.save({"state_dict": model.state_dict(), "threshold": threshold}, "rpw_autoencoder.pth")