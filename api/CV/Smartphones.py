import torch
import torch.nn as nn
from torchvision import transforms
import io
from PIL import Image
from fastapi import HTTPException, APIRouter, File, UploadFile, status

class Vgg16Logic(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256 * 16 * 16, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 5)
    )

  def forward(self, layer):
    layer = self.first(layer)
    return self.second(layer)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10)
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Vgg16Logic().to(device)
model.load_state_dict(torch.load('models/smartphones_model.pth', map_location=device))
model.eval()

labels = ['Google Pixel', 'Huawei', 'Iphone', 'Samsung', 'Xiaomi']

phones_router = APIRouter(prefix='/phones', tags=['CV'])

@phones_router.post('/')
async def predict_img(file: UploadFile = File(...)):
    try:
        img_data = await file.read()
        if not img_data:
            raise HTTPException(detail='Upload the image!', status_code=400)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        tensor_img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            predict = model(tensor_img)
            predict_index = predict.argmax(dim=1).item()
            predict_label = labels[predict_index]

            return {'Prediction': predict_label}
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)