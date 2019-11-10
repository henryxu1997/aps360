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

def load_sst_dataset(batch_size=32, device='cpu', root='.data'):
    """
    Creates BucketIterators for train, validation, and test sets from SST.
    Set device='cuda' to use GPU.
    """

    text_field = torchtext.data.Field(
        sequential=True, batch_first=True, include_lengths=True)
    label_field = torchtext.data.Field(
        sequential=False, batch_first=True)

    train, val, test = torchtext.datasets.SST.splits(
        text_field, label_field, root=root, fine_grained=True)

    text_field.build_vocab(train, vectors=glove)
    label_field.build_vocab(train)

    return torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size, device=device)


if __name__ == '__main__':
    test = split_text('the quick brown fox jumped over the green turtle. it was really exciting! HYPE?')
    embedding = convert_words_to_embeddings(test)
    print(embedding.shape, len(test))
