import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class BaselineOCR(nn.Module):
    """
    Simple model for classifying words belonging to a fixed number of classes.
    """
    def __init__(self, num_classes):
        super(BaselineOCR, self).__init__()
        # expected (32, 100) input
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.conv2 = nn.Conv2d(5, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # (((32 - 4) / 2 - 2) / 2) * (((100 - 4) / 2 - 2) / 2) * 10
        # 6 * 23 * 10 = 1380
        self.fc1 = nn.Linear(1380, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1380)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train/')
    valid_dir = os.path.join(data_dir, 'valid/')

    data_transform = transforms.Compose(
        [transforms.Resize((32, 100)),transforms.ToTensor()])

    train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transform)

    return train_data, valid_data

def evaluate(model, data, criterion):
    total_loss = 0.0
    loader = torch.utils.data.DataLoader(data, batch_size=64)
    for i, data in enumerate(loader):
        img, label = data
        out = model(img)
        loss = criterion(out, label)
        total_loss += loss.item()
    loss = float(total_loss) / (i + 1)
    return loss

def get_accuracy(model, data):
    correct = 0
    total = 0
    for imgs, labels in torch.utils.data.DataLoader(data, batch_size=64):
        output = model(imgs)
        
        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def train(model, train_data, valid_data, batch_size=64, learning_rate=0.001,
    num_epochs=1):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    iters, train_loss, train_acc, valid_loss, valid_acc = [], [], [], [], []

    # training
    n = 0
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_loader):
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iters.append(n)
            train_loss.append(float(loss)/batch_size)
            train_acc.append(get_accuracy(model, train_data))
            valid_loss.append(evaluate(model, valid_data, criterion))
            valid_acc.append(get_accuracy(model, valid_data))
            n += 1

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, train_loss, label="Train")
    plt.plot(iters, valid_loss, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, valid_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(valid_acc[-1]))

if __name__ == '__main__':
    data_dir = 'baseline_data'
    train_data, valid_data = load_data(data_dir)
    num_classes = len(train_data.classes)
    model = BaselineOCR(num_classes)
    train(model, train_data, valid_data, num_epochs=20)
