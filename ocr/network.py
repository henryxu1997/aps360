import os
import time

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms, datasets


class OCRNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # TODO(henry): define clearly what the input and output should be
        pass


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        self.name = 'CRNN'
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        print(b, c, h, w)
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output

def train_network(network, train_loader, val_loader=None, learning_rate=0.01, num_epochs=32):
    # Logging
    print('Number of parameters=', sum([param.numel() for param in network.parameters()]))
    print('Number of training batches =', len(train_loader))
    print(f'{network.name}:lr={learning_rate}:epochs={num_epochs}')

    # The loss function will be Cross Entropy Loss since multiclass classification.
    # Optimizer will be SGD with Momentum.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)

    # Zero initialize arrays which will be filled with loss and error values.
    train_loss, train_err = np.zeros(num_epochs), np.zeros(num_epochs)
    val_loss, val_err = np.zeros(num_epochs), np.zeros(num_epochs)

    start_time = time.time()
    for epoch in range(num_epochs):
        # Train
        per_epoch_train_loss = 0.
        per_epoch_train_error = 0
        for batch_idx, (batch, targets) in enumerate(train_loader):
            print(batch, targets)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # shape (batch_size, 9)
            outputs = network(batch)
            print(outputs)
            print(outputs.shape)

            # preds = outputs
            # _, preds = preds.max(2)
            # preds = preds.squeeze(2)
            # preds = preds.transpose(1, 0).contiguous().view(-1)
            # sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            # for pred, target in zip(sim_preds, cpu_texts):
            #     if pred == target.lower():
            #         n_correct += 1
            return

            # Can use raw target values here
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Add to running loss total
            per_epoch_train_loss += loss.item()

            # The index with maximum probability is the predicted label
            _, predicted = outputs.max(1)
            wrong = sum([1 if pred != actual else 0 for pred, actual in zip(predicted, targets)])
            per_epoch_train_error += wrong

        train_loss[epoch] = per_epoch_train_loss
        # Hardcoded the number of training examples used
        train_err[epoch] = per_epoch_train_error / 1452

        # Save the current model (checkpoint) to a file
        e_str = str(epoch).zfill(2)
        model_path = f'{network.name}:lr={learning_rate}:epoch={e_str}.pt'
        torch.save(network.state_dict(), model_path)

        if val_loader is None:
            print('Epoch=', epoch, ' train_err=', per_epoch_train_error, ' train_loss=', per_epoch_train_loss)
            continue
        # Validate
        per_epoch_val_loss = 0.
        per_epoch_val_error = 0
        for _, (batch, targets) in enumerate(val_loader):
            outputs = network(batch)
            loss = criterion(outputs, targets)
            per_epoch_val_loss += loss.item()
            _, predicted = outputs.max(1)
            wrong = sum([1 if pred != actual else 0 for pred, actual in zip(predicted, targets)])
            per_epoch_val_error += wrong
        val_loss[epoch] = per_epoch_val_loss
        # Hardcoded the number of validation examples used
        val_err[epoch] = per_epoch_val_error / 481
        print('Epoch=', epoch, ' train_err=', per_epoch_train_error, ' train_loss=', per_epoch_train_loss,
            'val_err', per_epoch_val_error, 'val_loss=', per_epoch_val_loss)

    print("Total time elapsed: {:.2f} seconds".format(time.time()-start_time))
    # Write the train/test loss/err into CSV file for plotting later
    csv_path = f'{network.name}:lr={learning_rate}:epochs={num_epochs}'
    np.savetxt("{}_train_loss.csv".format(csv_path), train_loss)
    np.savetxt("{}_train_err.csv".format(csv_path), train_err)
    if val_loader:
        np.savetxt('{}_val_loss.csv'.format(csv_path), val_loss)
        np.savetxt('{}_val_err.csv'.format(csv_path), val_err)
    return csv_path


def get_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]
    )
    import pytesseract
    from PIL import Image
    dataset = []
    greyscale = transforms.Grayscale(num_output_channels=1)

    for img_path in sorted(os.listdir('data'))[:10]:
        if img_path.endswith('.jpg'):
            img = Image.open(os.path.join('data', img_path))
            # Size found https://github.com/meijieru/crnn.pytorch/blob/master/dataset.py
            size = 100, 32
            resized_img = greyscale(img.resize(size))
            # resized_img.show()
            # IMPORTANT: Treat the image as a single word is psm 8
            # https://stackoverflow.com/questions/44619077/pytesseract-ocr-multiple-config-options
            label = pytesseract.image_to_string(img, config='--psm 8')
            if not label:
                img.show()
                raise ValueError('FUCK')
            tensor_img = transform(resized_img)
            print(label, tensor_img.shape)
            # print(tensor_img.shape)
            # Manually create dataset with example and label
            dataset.append((tensor_img, label))
    print(len(dataset))
    return DataLoader(dataset, batch_size=64)

def main():
    # Taken from defaults https://github.com/meijieru/crnn.pytorch
    img_h = 64
    nclass = len('0123456789abcdefghijklmnopqrstuvwxyz')
    nc = 1
    nh = 256 # size of hidden state

    crnn = CRNN(img_h, nc, nclass, nh)

    dataloader = get_data()
    print(crnn)


    train_network(crnn, dataloader)


if __name__ == '__main__':
    main()
