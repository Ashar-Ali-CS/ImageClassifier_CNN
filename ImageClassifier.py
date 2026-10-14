import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split




#----------- 1. Loading CIFAR-10, Preprocessing & Dataset Splitting------------------------------


#cifar10 contains - classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



torch.manual_seed(42)

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])

train_data = torchvision.datasets.CIFAR10(
    "./data", train=True, download=True, transform=train_transform
)
val_data = torchvision.datasets.CIFAR10(
    "./data", train=True, download=True, transform=eval_transform
)
test = torchvision.datasets.CIFAR10(
    "./data", train=False, download=True, transform=eval_transform
)

train_indices, val_indices = random_split(
    range(len(train_data)), [45000, 5000],
    generator=torch.Generator().manual_seed(42),
)
train = Subset(train_data, train_indices.indices)
val = Subset(val_data, val_indices.indices)

# Batch size
trainloader = DataLoader(train, batch_size=128, shuffle=True)
valloader = DataLoader(val, batch_size=128)
testloader = DataLoader(test, batch_size=128)


#----------- 2. Defining classification CNN model --------





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = CIFAR10CNN().to(device)







#----------- 3. Defining hyperparamters - epochs, Batch , loss  optimiser--------

EPOCH_COUNT = 5



loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_COUNT)
best_val_accuracy = 0.0
best_model_state = None



#----------- 4. TRAINING LOOP --------




for epoch in range(EPOCH_COUNT):
    model.train()
    running_loss = 0.0
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in valloader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)

    val_accuracy = correct / total
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
        }

    scheduler.step()
    print(
        f"Epoch {epoch + 1}/{EPOCH_COUNT}: "
        f"loss {running_loss / len(train):.4f}, "
        f"validation accuracy {100 * val_accuracy:.2f}%"
    )


#----------- 5. EVAL METRICS --------



model.load_state_dict(best_model_state)
model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)

print(f"Test accuracy: {100*correct/total:.2f}%")



