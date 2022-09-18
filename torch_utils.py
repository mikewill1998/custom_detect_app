import io
import torch 
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms 
import torchvision.transforms.functional as ttf
from PIL import Image

# load model

model = torchvision.models.resnet50(weights=None)
model.fc = nn.Linear(in_features=2048, out_features=3, bias=False)


in_channels = 3
num_classes = 3
PATH = "intel_screening_with_pretrained_resnet50.pth"
model.load_state_dict(torch.load(PATH, map_location=torch.device("cpu")))
model.eval()

# image -> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.CenterCrop((3000, 3000)),
        transforms.Resize((28,28)), transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), 
                            (0.24703233, 0.24348505, 0.26158768))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):
    images = image_tensor
    outputs = model(images)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
