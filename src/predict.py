import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(pretrained=False)

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 2)
)

model.load_state_dict(torch.load("../model.pth", map_location=device))
model = model.to(device)
model.eval()

# Transform (same as validation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Class labels
classes = ["Authentic", "Counterfeit"]

# 🔥 Function to predict
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return classes[predicted.item()], confidence.item()

# 🔥 Test it
if __name__ == "__main__":
    img_path = input("Enter image path: ")

    result, conf = predict_image(img_path)
    print(f"Prediction: {result}")
    print(f"Confidence: {conf*100:.2f}%")