
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_processing import load_sst_dataset, create_iter, split_text
from network import SANet

def get_accuracy(model, data_iter):
    correct, total = 0, 0
    for batch in data_iter:
        outputs = model(batch.text[0])
        output_prob = torch.softmax(outputs, dim=1)
        # indices in range 0-4 which is the same as batch.label
        _, indices = output_prob.max(1)
        # print(indices, batch.label)
        result = (indices == batch.label)
        correct += result.sum().item()
        total += len(result)
    return correct / total

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

def train_network(model, train_set, valid_set=None,
                  learning_rate=0.01, weight_decay=0.0, batch_size=64, num_epochs=32):
    """
    Customizable training loop
    """
    train_iter = create_iter(train_set, batch_size)
    if valid_set:
        valid_iter = create_iter(valid_set, batch_size)

    # Zero initialize arrays which will be filled with loss and error values.
    train_loss, train_acc = np.zeros(num_epochs), np.zeros(num_epochs)
    if valid_set:
        val_loss, val_acc = np.zeros(num_epochs), np.zeros(num_epochs)

    # Define loss and optimizer
    # Since multi-class classification, use CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        epoch_train_loss = 0.
        for batch in train_iter:
            outputs = model(batch.text[0])
            # Sanity check
            # assert outputs.shape == (batch_size, 5)
            loss = criterion(outputs, batch.label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_loss += loss.item()
        train_loss[epoch] = epoch_train_loss

        if valid_set:
            epoch_val_loss = 0.
            for batch in valid_iter:
                outputs = model(batch.text[0])
                loss = criterion(outputs, batch.label)
                epoch_val_loss += loss.item()
            val_loss[epoch] = epoch_val_loss

        # Get training, validation accuracy
        xx = get_accuracy(model, train_iter)
        train_acc[epoch] = xx

        if valid_set:
            yy = get_accuracy(model, valid_iter)
            val_acc[epoch] = yy
            print(f'Epoch {epoch}; Train loss {epoch_train_loss}; Val loss {epoch_val_loss}; Train acc {xx}; Val acc {yy}')
        else:
            print(f'Epoch {epoch}; Train loss {epoch_train_loss}; Train acc {xx}')
        if epoch % 10 == 0:
            e_str = str(epoch).zfill(2)
            model_path = f'{model.name}:lr={learning_rate}:wd={weight_decay}:b={batch_size}epoch={e_str}.pt'
            torch.save(model.state_dict(), os.path.join('models', model_path))

    csv_path = f'{model.name}:lr={learning_rate}:wd={weight_decay}:b={batch_size}:e={num_epochs}'
    np.savetxt("outputs/{}_train_loss.csv".format(csv_path), train_loss)
    np.savetxt("outputs/{}_train_acc.csv".format(csv_path), train_acc)
    if valid_set:
        np.savetxt('outputs/{}_val_loss.csv'.format(csv_path), val_loss)
        np.savetxt('outputs/{}_val_acc.csv'.format(csv_path), val_acc)
    return csv_path

def manual_run(sentence):
    words = split_text(sentence)
    from data_processing import glove
    model = SANet(glove.vectors)
    word_tensor = torch.zeros(len(words), dtype=int)
    print('Length of vocab', len(vocab))
    for i, word in enumerate(words):
        index = vocab.stoi[word]
        word_tensor[i] = index
    print(word_tensor)
    out = model(word_tensor.unsqueeze(0))
    print(out)


def make_dirs_if_not_exist():
    for directory in ['outputs', 'models', 'graphs']:
        if not os.path.exists(directory):
            os.makedirs(directory)
def main():
    make_dirs_if_not_exist()
    # For reproducibility, set a random seed
    torch.manual_seed(42)
    train_set, valid_set, test_set, vocab = load_sst_dataset()
    model = SANet(vocab.vectors)
    print(model)
    path = train_network(model, train_set, valid_set, num_epochs=64)
    plot_curves(path)


def call_with_options(char_base, three_labels, regression):
    make_dirs_if_not_exist()
    # For reproducibility, set a random seed
    torch.manual_seed(42)
    train_set, valid_set, test_set, vocab = load_sst_dataset( char_base = char_base, three_labels = three_labels, regression = regression)
    
    if char_base:
        model = CharNet()
    else:
        model = SANet(vocab.vectors)
    

    print(model)
    path = train_network(model, train_set, valid_set, num_epochs=64)
    plot_curves(path)




if __name__ == '__main__':
    # manual_run('the movie was phenomenal')
    main()
    char_base = False
    three_labels = False
    regression = False
