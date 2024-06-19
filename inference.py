import os

import torch
import torchvision.transforms as transforms

from data_model import ExampleDataset
from model import ExampleModel
from visualization import prepprocess_image, visualize_prediction

data_dir = "./model_dataset/train"
model_path = "./model/animal-MN-V4_best.pth"
test_image_path = "model_dataset/test/bat/image_9.png"
image_size = (128, 128)
num_classes = len(os.listdir(data_dir))


def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilites = torch.nn.functional.softmax(outputs, dim=1)
        return probabilites.cpu().numpy().flatten()


transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = ExampleModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()
dataset = ExampleDataset(data_dir)
orginal_image, image_tensor = prepprocess_image(test_image_path, transform)
probabilities = predict(model, image_tensor, device)
class_name = dataset.classes
visualize_prediction(orginal_image, probabilities, class_name)
