import os

import gradio as gr
import torch
import torchvision.transforms as transforms

from model import ExampleModel

model_path = "./model/animal-MN-V3_best.pth"
labels = [path for path in os.listdir("model_dataset/train/")]
num_classes = len(labels)
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ExampleModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()


def predict(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)
        prediction = torch.nn.functional.softmax(prediction, dim=1).squeeze()
        confidences = {labels[i]: float(prediction[i]) for i in range(num_classes)}
        return confidences


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=num_classes),
).launch()
