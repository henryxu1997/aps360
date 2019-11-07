import torch
import torchtext

EMBEDDING_SIZE = 50
glove = torchtext.vocab.GloVe(name="6B", dim=EMBEDDING_SIZE)


def convert_words_to_embeddings(words):
    """
    words = List[str]
    """
    print(type(glove[words[0]]))
    x = torch.zeros(size=[len(words), EMBEDDING_SIZE])
    for i, word in enumerate(words):
        x[i] = glove[word]
    print(x)
    return x

def load_sst_dataset():
    # TODO(ivan)
    sst = torchtext.datasets.SST()
    print(sst)


if __name__ == '__main__':
    test = 'the quick brown fox jumped over the green turtle'.split()
    convert_words_to_embeddings(test)
