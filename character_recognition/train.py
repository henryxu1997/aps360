import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from network import CharacterClassifier
from data_processing import load_dataset, get_small_dataloader, split_dataset, get_dataloader
import argparse
import string
from PIL import Image
import glob

def plot_curves(path, val=True):
    plots = {
        'Loss': ['train_loss', 'val_loss'],
        'Accuracy': ['train_acc', 'val_acc']
    }
    for plot_name, plots in plots.items():
        plt.title(f'Train vs Validation {plot_name}')
        train_nums = np.loadtxt(f'outputs/{path}_{plots[0]}.csv')
        n = len(train_nums)
        plt.plot(range(1,n+1), train_nums, label='Train')
        if val:
            val_nums = np.loadtxt(f'outputs/{path}_{plots[1]}.csv')
            plt.plot(range(1,n+1), val_nums, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(plot_name)
        plt.legend(loc='best')
        plt.savefig(f'graphs/{path}_{plot_name.lower()}.png')
        plt.show()

def get_accuracy(model, data_loader):
    correct, total = 0, 0
    for batch, targets in data_loader:
        outputs = model(batch)
        output_prob = torch.softmax(outputs, dim=1)
        _, indices = output_prob.max(1)
        # print(indices, batch.label)
        result = (indices == targets)
        correct += result.sum().item()
        total += len(result)
    return correct / total

def get_time_delta_str(start_time):
    return '{:.2f}'.format(time.time()-start_time)

def train_network(network, train_loader, val_loader=None, learning_rate=0.01, num_epochs=32):
    # Logging
    print('Number of parameters=', sum([param.numel() for param in network.parameters()]))
    print('Number of training batches =', len(train_loader))
    print(f'{network.name}:lr={learning_rate}:epochs={num_epochs}')

    # The loss function will be Cross Entropy Loss since multiclass classification.
    # Optimizer will be SGD with Momentum.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)

    # Zero initialize arrays which will be filled with loss and accuracy values.
    train_loss, train_acc = np.zeros(num_epochs), np.zeros(num_epochs)
    val_loss, val_acc = np.zeros(num_epochs), np.zeros(num_epochs)

    start_time = time.time()
    for epoch in range(num_epochs):
        # Train
        per_epoch_train_loss = 0.
        num_correct, num_examples = 0, 0
        for batch_idx, (batch, targets) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            outputs = network(batch)
            # Can use raw target values here
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # Add to running loss total
            per_epoch_train_loss += loss.item()

            # Put this here for training efficiency
            output_prob = torch.softmax(outputs, dim=1)
            _, indices = output_prob.max(1)
            num_correct += (indices == targets).sum().item()
            num_examples += len(targets)

        train_loss[epoch] = per_epoch_train_loss
        train_acc[epoch] = num_correct / num_examples #get_accuracy(network, train_loader)

        if epoch % 5 == 0:
            # Save the current model (checkpoint) to a file
            e_str = str(epoch).zfill(3)
            model_path = f'models/{network.name}:lr={learning_rate}:epoch={e_str}.pt'
            torch.save(network.state_dict(), model_path)

        if val_loader is None:
            print(f'Epoch={epoch} train_acc={train_acc[epoch]}, train_loss={train_loss[epoch]}, Time elapsed: {get_time_delta_str(start_time)}')
            continue
        # Validate
        per_epoch_val_loss = 0.
        num_correct, num_examples = 0, 0
        for _, (batch, targets) in enumerate(val_loader):
            outputs = network(batch)
            loss = criterion(outputs, targets)
            per_epoch_val_loss += loss.item()
            # Training efficiency
            output_prob = torch.softmax(outputs, dim=1)
            _, indices = output_prob.max(1)
            num_correct += (indices == targets).sum().item()
            num_examples += len(targets)
        val_loss[epoch] = per_epoch_val_loss
        val_acc[epoch] = num_correct / num_examples #get_accuracy(network, val_loader)
        print(f'Epoch={epoch} train_acc={train_acc[epoch]}, train_loss={train_loss[epoch]} val_acc {val_acc[epoch]}, val_loss={val_loss[epoch]} Time elapsed: {get_time_delta_str(start_time)}')

    print("Total time elapsed: {:.2f} seconds".format(time.time()-start_time))
    # Write the train/test loss/acc into CSV file for plotting later
    csv_path = f'{network.name}:lr={learning_rate}:epochs={num_epochs}'
    np.savetxt("outputs/{}_train_loss.csv".format(csv_path), train_loss)
    np.savetxt("outputs/{}_train_acc.csv".format(csv_path), train_acc)
    if val_loader:
        np.savetxt('outputs/{}_val_loss.csv'.format(csv_path), val_loss)
        np.savetxt('outputs/{}_val_acc.csv'.format(csv_path), val_acc)
    return csv_path
    

def verify_on_small_dataset():
    network = CharacterClassifier(num_classes=4)
    full_dataset = load_dataset()
    dataloader = get_small_dataloader(full_dataset, num_classes=4)
    path = train_network(network, dataloader)
    plot_curves(path, val=False)

def main():
    network = CharacterClassifier()
    full_dataset = load_dataset()
    train, val, test = split_dataset(full_dataset, batch_size=64)
    path = train_network(network, train, val, num_epochs=128)
    plot_curves(path)

def load_model(model_path):
    network = CharacterClassifier()
    network.load_state_dict(torch.load(model_path))
    return network

def test():
    # Chosen by Jordan as the best model
    model_path = 'models/nc=62:F=3:M=5:lr=0.01:epoch=010.pt'
    network = load_model(model_path)
    full_dataset = load_dataset()
    _, __, test = split_dataset(full_dataset)
    test_acc = get_accuracy(network, test)
    print(f'Test accuracy = {test_acc}')

def evaluate(input_folder, output_folder):
    character_order = string.digits + string.ascii_uppercase + string.ascii_lowercase
    model_path = 'models/nc=62:F=3:M=5:lr=0.01:epoch=010.pt'
    model = load_model(model_path)

    #Sort such that _10.jpg comes after _9.jpg
    images = sorted(glob.glob(os.path.abspath(input_folder) + "/*jpg"), key=lambda x: tuple(map(int,(x[:-4].split("_")[-1:-4:-1][::-1]))))
    output_f = open(output_folder + "/output.txt", 'w')
    currLine = []
    currWord = ""
    line = 1

    for img in images:
        img_name = os.path.splitext(os.path.basename(img))[0]
        temp = img_name.split("_")
        line_i, word_i, char_i = temp[-3], int(temp[-2]), temp[-1]
        
        img_f = Image.open(img).convert('RGB')
        img_np = np.asarray(img_f).transpose((2,0,1))
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()

        output = model(img_tensor)
        output_prob = torch.softmax(output, dim=1)
        _, indices = output_prob.max(1)
        char = character_order[indices[0]]

        if line != line_i:
            currLine.append(currWord)
            output_f.write(" ".join(currLine) + "\n")
            currWord = ""
            currLine = []
            line = line_i

        if word_i > 1 and word_i > len(currLine):
            currLine.append(currWord)
            currWord = char
        else:
            currWord += char
    
    #Last line
    currLine.append(currWord)
    output_f.write(" ".join(currLine) + "\n")
    
    output_f.close()

parser = argparse.ArgumentParser(description='Character Bounding Box Cropping')
parser.add_argument('--input_folder', default='../data/result/', type=str, help='folder path to input images')
parser.add_argument('--output_folder', default='../data/text/', type=str, help='folder path to results')
args = parser.parse_args()


if __name__ == '__main__':
    #verify_on_small_dataset()
    # main()
    evaluate(args.input_folder, args.output_folder)