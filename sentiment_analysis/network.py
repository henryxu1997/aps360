import torch
import torch.nn as nn

from data_processing import convert_words_to_embeddings, load_sst_dataset, create_iter


class Network(nn.Module):
    def __init__(self, vocab, hidden_size=10, output_size=5):
        """
        vocab_size = number of unique words in the context ?? or just length of embeding
        output_size= 5 (SST strong negative, negative, neutral, positive, strong positive)
        """
        super().__init__()
        self.name = f'Network'
        # TODO: build a simple network to start
        self.vocab_size, self.emb_size = vocab.vectors.shape

        # Create an embedding layer that will map a vector of word indices 
        # to embedding vectors of size emb_size.
        self.embed = nn.Embedding(self.vocab_size, self.emb_size)
        self.embed.weight.data.copy_(vocab.vectors)

        self.rnn_layer = nn.RNN(input_size=self.emb_size,
                                hidden_size=hidden_size,
                                batch_first=True)
        # Output - 2 classes spam or not spam
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x is Tensor of [batch_size, sentence_size, embedding_size]
        print(x.shape)
        x = self.embed(x)
        print(x.shape)
        out, _ = self.rnn_layer(x)
        out = torch.max(out, dim=1)[0]
        out = self.fc(out)
        return out


def train_network(model, train_set, valid_set, learning_rate=0.01, batch_size=64):
    train_iter = create_iter(train_set, batch_size)
    for batch in train_iter:
        outputs = model(batch.text[0])

        # Sanity check
        assert outputs.shape == (batch_size, 5)
        # Softmax along dim 1 (embedding dimension)
        output_prob = torch.softmax(outputs, dim=1)
        print(output_prob)
        break

def manual_run(model, sentence):
    xx = convert_words_to_embeddings(sentence.split())
    print(xx.shape)
    xx = xx.unsqueeze(0)
    print(xx.shape)
    out = model(xx)
    print(out)


def main():
    train_set, valid_set, test_set, vocab = load_sst_dataset()
    model = Network(vocab)
    train_network(model, train_set, valid_set)

if __name__ == '__main__':
    main()

