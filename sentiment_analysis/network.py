import torch
import torch.nn as nn


class SANet(nn.Module):
    """
    Customizable sentiment analysis neural network.
    """
    def __init__(self, embeddings, layer_type='rnn', hidden_size=10, num_layers=1, dropout=0.0):
        super().__init__()
        self.vocab_size, self.emb_size = embeddings.shape
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.name = f'SANet:{self.vocab_size}:{self.emb_size}:{layer_type}:{hidden_size}:{num_layers}:{dropout}'
        # output_size= 5 (strong negative, negative, neutral, positive, strong positive)
        self.output_size = 5    

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

    def forward(self, x):
        # x is Tensor of [batch_size, sentence_size]
        x = self.embed(x)
        # x is now Tensor of [batch_size, sentence_size, embedding_size]
        out, _ = self.rnn_layer(x)
        out = torch.max(out, dim=1)[0]
        out = self.fc(out)
        return out