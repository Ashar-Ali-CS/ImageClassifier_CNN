import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt




#----------- 1. Loading CIFAR-10, Preprocessing & Dataset Splitting------------------------------


#cifar10 contains - classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

data = torchvision.datasets.CIFAR10(
    "./data", train=True, download=True, transform=transform
)
test = torchvision.datasets.CIFAR10(
    "./data", train=False, download=True, transform=transform
)

train, val = random_split(data, [40000, 10000])
trainloader = DataLoader(train, batch_size=64, shuffle=True)
valloader = DataLoader(val, batch_size=64)
testloader = DataLoader(test, batch_size=64)






#----------- 2. Defining classification CNN model --------





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, 10)
).to(device)







#----------- 3. Defining hyperparamters - epochs, Batch , loss  optimiser--------

EPOCH_COUNT = 5



loss_fn = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters())



#----------- 4. TRAINING LOOP --------




for epoch in range(EPOCH_COUNT):
    model.train()
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in valloader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)

    print(f"Epoch {epoch+1}: validation accuracy {100*correct/total:.2f}%")


#----------- 5. EVAL METRICS --------



model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)

print(f"Test accuracy: {100*correct/total:.2f}%")



