import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_model import ExampleDataset
from model import ExampleModel

train_folder = "./model_dataset/train"
test_folder = "./model_dataset/test"
valid_folder = "./model_dataset/test"

num_classes = len(os.listdir(train_folder))
batch_size = 8
num_epoch = 21
train_losses, valid_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

train_dataset = ExampleDataset(train_folder, transform=transform)
val_dataset = ExampleDataset(valid_folder, transform=transform)
test_dataset = ExampleDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = ExampleModel(num_classes=num_classes)
model.to(device)

# params

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
valid_loss = 0.0
train_loss = 0.0
for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training loop"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # valid
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for valid_images, valid_labels in tqdm(val_loader, desc="Validation loop"):
            valid_images, valid_labels = valid_images.to(device), valid_labels.to(
                device
            )
            valid_outputs = model(valid_images)
            loss = criterion(valid_outputs, valid_labels)
            running_loss += loss.item() * valid_labels.size(0)
        valid_loss = running_loss / len(val_loader.dataset)
        valid_losses.append(valid_loss)

    print(
        f"Epoch : {epoch+1}/{num_epoch} - Train loss :{train_loss}, Validation loss: {valid_loss}"
    )

    os.makedirs("model", exist_ok=True)
    if epoch % 10 == 0:
        torch.save(obj=model.state_dict(), f=f"model/animal-MN-V3_{epoch}.pth")

    if min(valid_losses) >= valid_loss:
        torch.save(obj=model.state_dict(), f=f"model/animal-MN-V3_best.pth")
        print("Best model updated successfully")
    if min(train_losses) >= train_loss:
        torch.save(obj=model.state_dict(), f=f"model/animal-MN-V3_train_best.pth")
        print("Train Best model updated successfully")
