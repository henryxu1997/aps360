import torch
import torch.nn as nn

from data_processing import convert_words_to_embeddings

class Network(nn.Module):
    def __init__(self, vocab_size, hidden_size=10, output_size=5):
        """
        vocab_size = number of unique words in the context ?? or just length of embeding
        output_size= 5 (SST strong negative, negative, neutral, positive, strong positive)
        """
        super().__init__()
        self.name = f'Network'
        # TODO: build a simple network to start
        self.rnn_layer = nn.RNN(input_size=vocab_size,
                                hidden_size=hidden_size,
                                batch_first=True)
        # Output - 2 classes spam or not spam
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x is Tensor of [batch_size, sentence_size, embedding_size]
        out, _ = self.rnn_layer(x)
        out = torch.max(out, dim=1)[0]
        out = self.fc(out)
        return out


def train_network(model, train_set, valid_set, learning_rate=0.01, batch_size=64):
    pass

def manual_run(model, sentence):
    xx = convert_words_to_embeddings(sentence.split())
    print(xx.shape)
    xx = xx.unsqueeze(0)
    print(xx.shape)
    out = model(xx)
    print(out)



if __name__ == '__main__':
    model = Network(50)
    manual_run(model, 'the horse is brown')

