import torch
import torch.nn as nn
from torchaudio import transforms
import io
import soundfile as sf
from fastapi import File, UploadFile, APIRouter, HTTPException
import torch.nn.functional as F

class AudioLogic(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((8, 8))
    )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 8)
    )

  def forward(self, x):
    x = x.unsqueeze(1)
    x = self.first(x)
    return self.second(x)

transform = transforms.MelSpectrogram(
    sample_rate=48000,
    n_mels=40
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AudioLogic().to(device)
model.load_state_dict(torch.load('models/emotional_speech_model.pth', map_location=device))
model.eval()

labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

indx_to_lbl = {indx: lbl for indx, lbl in enumerate(labels)}

max_len = 1000
def change_audio(waveform):
    waveform = torch.tensor(waveform).T
    spec = transform(waveform).squeeze(0)

    spec_length = spec.shape[1]
    if spec_length > max_len:
        spec = spec[:, :max_len]

    if spec_length < max_len:
        spec = F.pad(spec, (0, max_len - spec_length))

    return spec

emotional_speech_router = APIRouter(prefix='/emotional_speech', tags=['SR'])

@emotional_speech_router.post('/')
async def predict_sound(file: UploadFile = File(...)):
    try:
        sound_data = await file.read()
        if not sound_data:
            raise HTTPException(detail='Upload the file!', status_code=400)
        waveform, sample_rate = sf.read(io.BytesIO(sound_data), dtype='float32')
        spec = change_audio(waveform).unsqueeze(0).to(device)

        with torch.no_grad():
            predict = model(spec)
            pred_indx = torch.argmax(predict, dim=1).item()
            pred_lbl = indx_to_lbl[pred_indx]

            return {
                'index': pred_indx,
                'label': pred_lbl
            }

    except Exception as e: raise HTTPException(detail=str(e), status_code=500)