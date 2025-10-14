# Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch-Completion-requirement


## AIM:

 To Build-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch-Completion-requirement.

 ## CODE:

 # Image Classification: Cats vs Dogs vs Pandas (Transfer Learning)
```
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import random


# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 1. DATA PREPARATION
# Paths to dataset folders (update if needed)
train_dir = 'data/train'
test_dir  = 'data/test'

# Image transforms
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
BATCH_SIZE = 32

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transforms = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE*1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset   = datasets.ImageFolder(test_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = train_dataset.classes

# 2. MODEL DESIGN (Transfer Learning: ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final classifier
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(classes))
)
model = model.to(device)

# 3. TRAINING
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

EPOCHS = 10  # for demo; can increase to 12-15
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
    
    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

# 4. EVALUATION
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

accuracy = total_correct / total_samples
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# 5. BONUS - Single Image Prediction


from PIL import Image

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = val_transforms
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    print(f"Prediction: {classes[pred.item()]}")
    print(f"Confidence: {conf.item()*100:.2f}%")
    return classes[pred.item()]
```

## OUTPUT:

<img width="353" height="266" alt="image" src="https://github.com/user-attachments/assets/048a1e16-1094-451a-bce6-701da102313c" />




## RESULT:

-An -AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch-Completion-requirement is executed sucessfully.


