import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, models, transforms
from tqdm import tqdm

"""
Code References:
    - https://www.kaggle.com/code/stpeteishii/guava-fruit-disease-fit-predict-lightning-w-tpu
    - https://www.kaggle.com/code/atharvcode/guava-fruit-disease-pytorch
    - https://www.kaggle.com/code/nawazhaider/resnet50-acc-100-clear-data-leakage-in-dataset#Resizing-and-Normalization-Layer
"""
"""
declaring device, batch size, image size, and epoch size based on common values.
Note: 
    - somehow i find 224 as common number on computer vision, however still need to dig deeper.
    - somehow i find 32 and 50 a common number, i dunno why. but i will find out
"""

batch_size = 32
image_size = 224
epoch_size = 50
data_class = 3
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

"""
transforming image data for preprocessing 
"""

train_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(224),
        transforms.transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

"""
locating dataset
"""

train_ds = datasets.ImageFolder(root="datasets/train", transform=transform)
val_ds = datasets.ImageFolder(root="datasets/val", transform=test_transform)
test_ds = datasets.ImageFolder(root="datasets/test", transform=test_transform)
print(f"Number of samples train: {len(train_ds)}")
print(f"Number of samples test: {len(val_ds)}")
print(f"Number of samples val: {len(test_ds)}")


"""
loading dataset
"""

train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_dl = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
val_dl = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Importing squeezenet as base model and freezing parameter
model = models.squeezenet1_1(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Fine tuning based model by changing convolutional layer
num_features = model.classifier[1].in_channels
model.classifier[1] = nn.Conv2d(
    in_channels=num_features, out_channels=data_class, kernel_size=(1, 1), stride=(1, 1)
)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epoch_size):
    model.train()
    for inputs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epoch_size}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluating Model
model.eval()
total = 0
correct = 0
running_loss = 0.0

with torch.no_grad():
    for inputs, labels in val_dl:
        inputs, labels = inputs.to(device), outputs.to(device)
        outputs = model(inputs)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

average_loss = running_loss / len(val_dl)
accuracy = 100 * correct / total

print(f"Validation Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

# Saving the model
torch.save(model.state_dict(), "model.pth")
