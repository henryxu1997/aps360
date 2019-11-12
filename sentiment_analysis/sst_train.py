import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import numpy as np
import matplotlib.pyplot as plt

from data_processing import get_iters,load_sst_dataset

root = ''


glove = torchtext.vocab.GloVe(name="6B", dim=50)

def split_sentence(line):
    line = line.replace(".", " . ").replace(",", " , ").replace(";", " ; ").replace("?", " ? ").lower()
    words = line.split()
    return words


class WordRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(WordRNN, self).__init__()
        self.num_layers = 2
        self.emb = nn.Embedding.from_pretrained(glove.vectors)
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size,hidden_size,num_layers = self.num_layers, batch_first=True, dropout = 0.4)
        self.fc = nn.Linear(hidden_size*2,5)

    def forward(self,x):
        x = self.emb(x)  # given batches of sentences
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)
        out,_ = self.rnn(x,h0)
        out = torch.cat([torch.max(out, dim=1)[0], 
                torch.mean(out, dim=1)], dim=1)
        out = self.fc(out)
        return out

# binary classification

def get_accuracy(model, data_iter):
    """ Compute the accuracy of the `model` across a dataset `data`
    
    Example usage:
    
    >>> model = MyRNN() # to be defined
    >>> get_accuracy(model, valid) # the variable `valid` is from above
    """

    acc = 0.0
    num_iter = 0
    for batch in data_iter:
        # batch contains sms, length of sms, and spam or not
        label = batch.label
        output = model(batch.text[0])
        #output = torch.sigmoid(output)

        pred = output.max(1, keepdim=True)[1]
        acc+=pred.eq(label.view_as(pred)).sum().item()
        
        num_iter += len(label)
    
    acc = float(acc)/float(num_iter)
    return acc

def get_loss(model,data_iter):
    criterion = nn.CrossEntropyLoss()
    tot_loss = 0.0
    for i,batch in enumerate(data_iter):
        label = batch.label
        output = model(batch.text[0])
        loss = criterion(output,label)
        tot_loss += float(loss)
    return tot_loss/(i+1)

train_text,val_text,test_text,vocab = load_sst_dataset()

def train_model(model, batch_size = 20, learning_rate = 0.01, num_epoch = 5,
                weight_decay = 0.0):
      

    train_iter = torchtext.data.BucketIterator(train_text,
                                           batch_size=batch_size,
                                           sort_key=lambda x: len(x.text), # to minimize padding
                                           sort_within_batch=True,        # sort within each batch
                                           repeat=False)                  # repeat the iterator for many epochs
    valid_iter = torchtext.data.BucketIterator(val_text,
                                           batch_size=batch_size,
                                           sort_key=lambda x: len(x.text), # to minimize padding
                                           sort_within_batch=True,        # sort within each batch
                                           repeat=False)                  # repeat the iterator for many epochs
    
    #train_iter,valid_iter,test_iter = get_iters(batch_size = batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay = weight_decay)

    train_acc,train_loss,val_acc,val_loss = [],[],[],[]    
    

    for epoch in range(num_epoch):
        total_t_loss = 0.0
        for i,batch in enumerate(train_iter):
            label = batch.label
            data = batch.text[0]
            
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_t_loss += float(loss)

        train_loss.append(total_t_loss/(i+1))
        train_acc.append(get_accuracy(model,train_iter))
        val_loss.append(get_loss(model,valid_iter))
        val_acc.append(get_accuracy(model, valid_iter))

        print(("Epoch {}: Train accuracy: {}, Train loss: {} |"+
                   "Validation accuracy: {}, Validation loss: {}").format(
                       epoch + 1,
                       train_acc[epoch],
                       train_loss[epoch],
                       val_acc[epoch],
                       val_loss[epoch]))

    np_val_acc = np.array(val_acc)
    i = np.argmax(np_val_acc)
    print(i+1,np_val_acc[i])
    #plot curve
    epochs = range(len(train_loss))
    
    plt.title("Loss Functions")
    plt.plot(epochs,train_loss, label = "train")
    plt.plot(epochs,val_loss, label = "validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    plt.savefig('loss.png')
    plt.close()
    plt.title("Accuracy")
    plt.plot(epochs,train_acc, label = "train")
    plt.plot(epochs,val_acc, label = "validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()
    plt.savefig('accuracy.png')

model = WordRNN(50,50)
train_model(model,batch_size = 100, learning_rate = 0.001, num_epoch = 300,weight_decay = 0.001)