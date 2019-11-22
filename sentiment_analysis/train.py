import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_processing import load_sst_dataset, create_iter, split_text
from network import WordSANet, CharSANet

def get_regression_accuracy(model, data_iter, three_labels):
    correct, total = 0, 0
    for batch in data_iter:
        outputs = model(batch.text[0])

        for (output,label) in zip(outputs,batch.label):
            if three_labels:
                if output < 0.5 and label == 0:
                    correct+=1
                elif output < 1.5 and label == 1:
                    correct+=1
                elif output>=1.5 and label == 2:
                    correct+=1
            else:
                if output < 0.5 and label == 0:
                    correct+=1
                elif output < 1.5 and label == 1:
                    correct+=1
                elif output < 2.5 and label == 2:
                    correct+=1
                elif output < 3.5 and label == 3:
                    correct+=1
                elif output>=3.5 and label == 4:
                    correct+=1
            total+=1
    return correct/total


def get_accuracy(model, data_iter):
    correct, total = 0, 0
    model.eval()
    for batch in data_iter:
        outputs = model(batch.text[0])
        output_prob = torch.softmax(outputs, dim=1)
        # indices in range 0-4 (5 classes) or 0-2 (3 classes) which is the
        # same as batch.label
        _, indices = output_prob.max(1)
        # print(indices, batch.label)
        result = (indices == batch.label)
        correct += result.sum().item()
        total += len(result)
    model.train()
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
        plt.close()

def train_network(model, train_set, valid_set=None, regression=False, three_labels=False,
                  learning_rate=0.01, weight_decay=0.0, batch_size=64, num_epochs=32,
                  val_acc_target=None):
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
    if not regression:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        epoch_train_loss = 0.
        for i,batch in enumerate(train_iter):
            outputs = model(batch.text[0])
            # Sanity check
            # assert outputs.shape == (batch_size, 5)
            loss = criterion(outputs, batch.label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_train_loss += loss.item()
        epoch_train_loss = epoch_train_loss/(i+1)
        train_loss[epoch] = epoch_train_loss

        if valid_set:
            epoch_val_loss = 0.
            for i,batch in enumerate(valid_iter):
                outputs = model(batch.text[0])
                loss = criterion(outputs, batch.label)
                epoch_val_loss += loss.item()
            epoch_val_loss = epoch_val_loss/(i+1)
            val_loss[epoch] = epoch_val_loss

        # Get training, validation accuracy
        if regression:
            xx = get_regression_accuracy(model,train_iter,three_labels)
        else:
            xx = get_accuracy(model, train_iter)
        train_acc[epoch] = xx

        if valid_set:
            if regression:
                yy = get_regression_accuracy(model, valid_iter,three_labels)
            else:
                yy = get_accuracy(model, valid_iter)
            val_acc[epoch] = yy
            print(f'Epoch {epoch}; Train loss {epoch_train_loss}; Val loss {epoch_val_loss}; Train acc {xx}; Val acc {yy}')
        else:
            print(f'Epoch {epoch}; Train loss {epoch_train_loss}; Train acc {xx}')
        if epoch % 10 == 0 or (valid_set is not None and
            val_acc_target is not None and yy >= val_acc_target):
            e_str = str(epoch).zfill(2)
            model_path = f'{model.name}:lr={learning_rate}:wd={weight_decay}:b={batch_size}epoch={e_str}.pt'
            torch.save(model.state_dict(), os.path.join('models', model_path))
    if valid_set:
        np_val_acc = np.array(val_acc)
        i = np.argmax(np_val_acc)
        print(i+1,np_val_acc[i])
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
    train_set, valid_set, test_set, vocab = load_sst_dataset(
        char_base=char_base, three_labels=three_labels, regression=regression)
    
    output_size = 5
    if three_labels:
        output_size = 3
    if regression:
        output_size = 1

    if char_base:
        model = CharSANet(vocab, layer_type='rnn', output_size=output_size,
            regression=regression)
    else:
        model = WordSANet(vocab.vectors, layer_type='lstm',
            output_size=output_size,regression=regression, hidden_size=108,
            num_layers=1, dropout=0.0)
    
    print(model)
    path = train_network(model, train_set, valid_set,
        three_labels=three_labels, regression=regression, num_epochs=30,
        learning_rate=0.0007, batch_size=64, val_acc_target=0.66)
    plot_curves(path)

    test_accuracy(test_set, model, regression, three_labels)

def test_accuracy(test_set, model, regression, three_labels):
    test_iter = create_iter(test_set, 64)
    if regression:
        test_acc = get_regression_accuracy(model, test_iter, three_labels)
    else:
        test_acc = get_accuracy(model, test_iter)
    print('Test accuracy:', test_acc)

def saved_model_test_accuracy(saved_model_file):
    _, _, test_set, vocab = load_sst_dataset(
        char_base=char_base, three_labels=three_labels, regression=regression)

    model = WordSANet(vocab.vectors, layer_type='lstm',
        output_size=3,regression=False, hidden_size=108,
        num_layers=1, dropout=0.0)
    model.load_state_dict(torch.load(saved_model_file))
    model.eval()

    test_accuracy(test_set, model, False, True)

if __name__ == '__main__':
    # manual_run('the movie was phenomenal')
    # main()

    char_base = False
    three_labels = True
    regression = False
    call_with_options(char_base=char_base, three_labels=three_labels, regression=regression)

    # saved_model_file = './saved_66.9p_epoch10/saved_models/WordSANet:16531:200:lstm:108:1:0.0:lr=0.0007:wd=0.0:b=64epoch=10.pt'
    # saved_model_test_accuracy(saved_model_file)