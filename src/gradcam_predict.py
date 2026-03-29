import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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

# Target layer (IMPORTANT)
target_layers = [model.layer4[-1]]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

classes = ["Authentic", "Counterfeit"]

# Prediction + GradCAM
def predict_with_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    rgb_img = np.array(image.resize((224, 224))) / 255.0

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    # Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    # 🔥 Divide heatmap into regions
    h, w = grayscale_cam.shape

    top = grayscale_cam[:h//3, :]
    middle = grayscale_cam[h//3:2*h//3, :]
    bottom = grayscale_cam[2*h//3:, :]

# Compute intensity per region
    top_score = np.mean(top)
    middle_score = np.mean(middle)
    bottom_score = np.mean(bottom)
    regions = []

    if top_score > 0.4:
        regions.append("upper area (collar/logo)")

    if middle_score > 0.4:
        regions.append("middle area (side/logo)")

    if bottom_score > 0.4:
        regions.append("lower area (toe box/sole)")
    
    if predicted.item() == 0:
         explanation = f"Model focused on {', '.join(regions)} showing consistent authentic features"
    else:
        explanation = f"Model focused on {', '.join(regions)} indicating possible defects (stitching/shape/logo issues)"
    if len(regions) == 0:
        explanation = "Model focus is unclear → low confidence decision"

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # 🔥 Simple explanation logic
    heatmap_intensity = np.mean(grayscale_cam)

    if heatmap_intensity > 0.5:
        explanation = "Model strongly focused on key features like stitching/logo"
    elif heatmap_intensity > 0.3:
        explanation = "Model moderately focused on important regions"
    else:
        explanation = "Model focus is weak → prediction may be less reliable"

    return classes[predicted.item()], confidence.item(), visualization, explanation


if __name__ == "__main__":
    img_path = input("Enter image path: ")

    result, conf, cam_img, explanation = predict_with_gradcam(img_path)

    print(f"Prediction: {result}")
    print(f"Confidence: {conf*100:.2f}%")
    print(f"Explanation: {explanation}")


    # Show heatmap
    cv2.imshow("Grad-CAM", cam_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

