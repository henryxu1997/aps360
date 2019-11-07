import torch.nn as nn

from data_processing import convert_words_to_embeddings


class Network(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x is Tensor of [batch_size, sentence_size, embedding_size]
        pass




def train_network():
    model = Network()
    convert_words_to_embeddings('train network'.split())

if __name__ == '__main__':
    train_network()

