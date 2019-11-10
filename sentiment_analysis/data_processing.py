import torch
import torchtext

EMBEDDING_SIZE = 50
glove = torchtext.vocab.GloVe(name="6B", dim=EMBEDDING_SIZE)

def split_text(text):
    # Utility method to separate punctuation, split and convert words to lower case.
    text = text.replace(".", " . ") \
                .replace(",", " , ") \
                .replace(";", " ; ") \
                .replace("?", " ? ") \
                .replace("!", " ! ")
    return text.lower().split()

def convert_words_to_embeddings(words):
    """
    words = List[str]
    """
    x = torch.zeros(size=[len(words), EMBEDDING_SIZE])
    for i, word in enumerate(words):
        x[i] = glove[word]
    return x

def load_sst_dataset():
    # TODO(ivan)
    sst = torchtext.datasets.SST()
    print(sst)


if __name__ == '__main__':
    test = split_text('the quick brown fox jumped over the green turtle. it was really exciting! HYPE?')
    embedding = convert_words_to_embeddings(test)
    print(embedding.shape, len(test))
