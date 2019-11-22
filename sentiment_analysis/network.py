import torch
import torch.nn as nn
import torch.nn.functional as F

# defined in chid: embedding/vector, name

class baseSANet(nn.Module):
    def __init__(self, layer_type='rnn', hidden_size=10, num_layers=1, dropout=0.0, output_size = 5,regression = False):
        super().__init__()
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        if layer_type == 'rnn':
            self.rnn_layer = nn.RNN(input_size=self.input_size,
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    dropout=dropout, 
                                    batch_first=True)
        elif layer_type == 'gru':
            self.rnn_layer = nn.GRU(input_size=self.input_size, 
                                    #input_size = 100,
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    dropout=dropout, 
                                    batch_first=True,
                                    bidirectional=True)
        elif layer_type == 'lstm':
            self.rnn_layer = nn.LSTM(input_size=self.input_size, 
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    dropout=dropout, 
                                    batch_first=True)
        else:
            raise ValueError(f'Invalid layer_type {layer_type}')
        # Define fully connected layer
        #self.norm_layer = LayerNorm(hidden_size, learnable=True)
        self.fc1 = nn.Linear(hidden_size*2, 30) 
        self.fc2 = nn.Linear(30, self.output_size)
        self.preprocess1 = nn.Linear(self.input_size, 100)
        #self.preprocess2 = nn.Linear(500,self.input_size)
        self.regression = regression
        self.dropout = nn.Dropout(0.4)


    def forward(self, x):
        # x is Tensor of [batch_size, sentence_size]
        #out = self.preprocess1(x)
        #out = self.preprocess2(out)
        out, _ = self.rnn_layer(x)
        out = torch.max(out, dim=1)[0]
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        if self.regression:
            out = out.squeeze(1)
        return out

class WordSANet(baseSANet):
    """
    Customizable sentiment analysis neural network.
    """
    def __init__(self, embeddings, layer_type='rnn', hidden_size=10, num_layers=1, dropout=0.0, output_size = 5,regression = False):
        
        self.vocab_size, self.emb_size = embeddings.shape
        self.input_size = self.emb_size
        self.name = f'WordSANet:{self.vocab_size}:{self.emb_size}:{layer_type}:{hidden_size}:{num_layers}:{dropout}'
        super().__init__(layer_type, hidden_size, num_layers, dropout, output_size, regression)
        self.embed = nn.Embedding.from_pretrained(embeddings)
        

    def forward(self, x):
        # x is Tensor of [batch_size, sentence_size]
        x = self.embed(x)
        return super().forward(x)

class CharSANet(baseSANet):
    def __init__(self, vocab, layer_type='rnn', hidden_size=10, num_layers=1, dropout=0.0, output_size = 5,regression = False):
        self.one_hot_dict = []
        length = len(vocab.itos)
        identity = torch.eye(length)
        for i in range(length):
            self.one_hot_dict.append(identity[i])

        #self.one_hot_dict = identity might do the same
        #TODO: try this later
        self.input_size = length
        self.name = f'CharSANet:{self.input_size}:{self.input_size}:{layer_type}:{hidden_size}:{num_layers}:{dropout}'

        super().__init__(layer_type = layer_type, hidden_size = hidden_size, num_layers = num_layers,\
                                 dropout = dropout, output_size = output_size, regression = regression)
        

    def char_to_one_hot(self,x):
        return torch.stack([torch.stack([self.one_hot_dict[char] for char in sentence]) \
                 for sentence in x])
    def forward(self,x):
        x = self.char_to_one_hot(x)
        return super().forward(x)

class test_ann(nn.Module):
    def __init__(self,embeddings, input_size, output_size):
        super().__init__()  
        self.embed = nn.Embedding.from_pretrained(embeddings)
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 300)
        self.activation = F.relu
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100,output_size)
    def forward(self, x):
        x = self.embed(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.output_size)
        return x

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = F.sigmoid()

        #define weights
        #U is the weight matrix for weights between input and hidden layers
        #V is the weight matrix for weights between hidden and output layers
        #W is the weight matrix for shared weights in the RNN layer (hidden layer)
       # self.U = 
    def forward(self,x):
        pass


