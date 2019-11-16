import os
import string
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from network import CharacterClassifier
from data_processing import load_dataset, get_small_dataloader, split_dataset
'''
def rename_data():
    # Character data from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
    character_order = string.digits + string.ascii_uppercase + string.ascii_lowercase
    print(character_order)
    dir_names = sorted(os.listdir('data'))
    print(len(character_order), len(dir_names))
    assert len(dir_names) == len(character_order)

    for c, dir_name in zip(character_order, dir_names):
        if dir_name.startswith('Sample'):
            old_path = os.path.join('data', dir_name)
            if c in string.digits:
                os.rename(old_path, os.path.join('data', c + '_digit'))
            elif c in string.ascii_uppercase:
                os.rename(old_path, os.path.join('data', c + '_upper'))
            if c in string.ascii_lowercase:
                os.rename(os.path.join('data', dir_name), os.path.join('data', c + '_lower'))
rename_data()
'''
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
        num_correct = 0
        num_examples = 0
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

        train_loss[epoch] = per_epoch_train_loss
        train_acc[epoch] = get_accuracy(network, train_loader)

        # Save the current model (checkpoint) to a file
        e_str = str(epoch).zfill(3)
        model_path = f'models/{network.name}:lr={learning_rate}:epoch={e_str}.pt'
        torch.save(network.state_dict(), model_path)

        if val_loader is None:
            print('Epoch=', epoch, ' train_acc=', train_acc[epoch], ' train_loss=', train_loss[epoch])
            continue
        # Validate
        per_epoch_val_loss = 0.
        for _, (batch, targets) in enumerate(val_loader):
            outputs = network(batch)
            loss = criterion(outputs, targets)
            per_epoch_val_loss += loss.item()
        val_loss[epoch] = per_epoch_val_loss
        val_acc[epoch] = get_accuracy(network, val_loader)
        print('Epoch=', epoch, ' train_acc=', train_acc[epoch], ' train_loss=', train_loss[epoch], 
            'val_acc', val_acc[epoch], 'val_loss=', val_loss[epoch])

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
    # network = CharacterClassifier(num_classes=4)
    # full_dataset = load_dataset()
    # dataloader = get_small_dataloader(full_dataset, num_classes=4)
    # path = train_network(network, dataloader)
    path = 'nc=4:F=3:M=5:lr=0.01:epochs=32'
    plot_curves(path, val=False)

def main():
    network = CharacterClassifier()
    full_dataset = load_dataset()
    train, val, test = split_dataset(full_dataset, batch_size=64)
    path = train_network(network, train, val, num_epochs=128)
    plot_curves(path)

if __name__ == '__main__':
    main()