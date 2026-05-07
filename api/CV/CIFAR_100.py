import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from fastapi import HTTPException, File, UploadFile, APIRouter

class VggCIFAR100(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d(2),
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

mean = [0.5071, 0.4867, 0.4408]
std  = [0.2675, 0.2565, 0.2761]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VggCIFAR100().to(device)
model.load_state_dict(torch.load('models/cifar_100_model.pth', map_location=device))
model.eval()

labels = torch.load('models/labels/cifar_100_labels.pth')

cifar_100_router = APIRouter(prefix='/cifar-100', tags=['CV'])

@cifar_100_router.post('/')
async def predict_img(file: UploadFile = File(...)):
    try:
        img_data = await file.read()
        if not img_data:
            raise HTTPException(detail='Upload the file!', status_code=400)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        tensor_img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            predict = model(tensor_img)
            pred = predict.argmax(dim=1).item()

            return {'Prediction': labels[pred]}

    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)