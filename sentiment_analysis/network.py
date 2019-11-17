import torch
import torch.nn as nn

class SANet(nn.Module):
    """
    Customizable sentiment analysis neural network.
    """
    def __init__(self, embeddings, layer_type='rnn', hidden_size=10, num_layers=1, dropout=0.0, output_size = 5,regression = False):
        super().__init__()
        if not isinstance(self,CharSANet):
            self.vocab_size, self.emb_size = embeddings.shape
        else:
            self.vocab_size = self.emb_size = len(embeddings.itos)
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.name = f'SANet:{self.vocab_size}:{self.emb_size}:{layer_type}:{hidden_size}:{num_layers}:{dropout}'
        # output_size= 5 (strong negative, negative, neutral, positive, strong positive)
        self.output_size = output_size

        # Create an embedding layer that will map a vector of word indices 
        # to embedding vectors of size emb_size.
        # TODO: verify validity of embedding
        # self.embed = nn.Embedding(self.vocab_size, self.emb_size)
        # self.embed.weight.data.copy_(embeddings)
        self.embed = nn.Embedding.from_pretrained(embeddings)

        if layer_type == 'rnn':
            self.rnn_layer = nn.RNN(input_size=self.emb_size,
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    dropout=dropout, 
                                    batch_first=True)
        elif layer_type == 'gru':
            self.rnn_layer = nn.GRU(input_size=self.emb_size, 
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    dropout=dropout, 
                                    batch_first=True)
        else:
            raise ValueError(f'Invalid layer_type {layer_type}')
        # Define fully connected layer
        self.fc = nn.Linear(hidden_size, self.output_size)
        self.regression = regression

    def forward(self, x):
        # x is Tensor of [batch_size, sentence_size]
        if not isinstance(self,CharSANet):
            x = self.embed(x)
        else:
            x = self.char_to_one_hot(x)
        # x is now Tensor of [batch_size, sentence_size, embedding_size]
        out, _ = self.rnn_layer(x)
        out = torch.max(out, dim=1)[0]
        out = self.fc(out)
        if self.regression:
            out = out.squeeze(1)
        return out

class CharSANet(SANet):
    def __init__(self, embeddings, layer_type='rnn', hidden_size=10, num_layers=1, dropout=0.0, regression = False):
        super(CharSANet, self).__init__(embeddings = embeddings, layer_type = layer_type,\
                                 hidden_size = hidden_size, num_layers = num_layers, dropout = dropout, regression = regression)
        self.one_hot_dict = []
        length = self.vocab_size
        identity = torch.eye(length)

        self.name = f'CharSANet:{self.vocab_size}:{self.emb_size}:{layer_type}:{hidden_size}:{num_layers}:{dropout}'


        #self.one_hot_dict = identity might do the same
        #TODO: try this later

        for i in range(length):
            self.one_hot_dict.append(identity[i])


        self.char_to_one_hot


    def char_to_one_hot(self,x):
        for sentence in x:
            for char in sentence:
                if char>self.emb_size:
                    print(char)

        return torch.stack([torch.stack([self.one_hot_dict[char] for char in sentence]) \
                 for sentence in x])