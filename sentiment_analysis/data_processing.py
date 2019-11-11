import os

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

    def get_label_value(label):
        return {'0': 0.0, '1': 0.25, '2': 0.5, '3': 0.75, '4': 1.0,
            None: None}[label]

    text_field = torchtext.data.Field(
        sequential=True, batch_first=True, include_lengths=True)
    label_field = torchtext.data.Field(
        sequential=False, batch_first=True, use_vocab=False,
        dtype=torch.float32,
        preprocessing=torchtext.data.Pipeline(get_label_value))

    fields = [('text', text_field), ('label', label_field)]

    path = torchtext.datasets.SST.download(root)

    train_file = 'train.txt'
    valid_file = 'dev.txt'
    test_file = 'test.txt'

    train_path = os.path.join(path, train_file)
    valid_path = os.path.join(path, valid_file)
    test_path = os.path.join(path, test_file)

    with open(os.path.expanduser(train_path)) as f:
        train_examples = [torchtext.data.Example.fromtree(line, fields) for line in f]
    with open(os.path.expanduser(valid_path)) as f:
        valid_examples = [torchtext.data.Example.fromtree(line, fields) for line in f]
    with open(os.path.expanduser(test_path)) as f:
        test_examples = [torchtext.data.Example.fromtree(line, fields) for line in f]

    train_data = torchtext.data.Dataset(train_examples, fields)
    valid_data = torchtext.data.Dataset(valid_examples, fields)
    test_data = torchtext.data.Dataset(test_examples, fields)

    text_field.build_vocab(train_data, vectors=glove)

    return torchtext.data.BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size, device=device)


if __name__ == '__main__':
    test = split_text('the quick brown fox jumped over the green turtle. it was really exciting! HYPE?')
    embedding = convert_words_to_embeddings(test)
    print(embedding.shape, len(test))

    # train_iter, valid_iter, test_iter = load_sst_dataset()
    # for batch in train_iter:
    #     print(batch.text)
    #     print(batch.label)
