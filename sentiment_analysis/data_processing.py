from collections import defaultdict
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

def _analyze_data(dataset):
    """Prints out an example from every category (0-4)."""
    wanted = 0
    counts = defaultdict(int)
    for example in dataset:
        counts[example.label] += 1
        if example.label == wanted:
            wanted += 1
            print('Example:', example.text, '\nLabel:', example.label)
    print('Label counts:', sorted(counts.items()))
    
def load_sst_dataset(root='.data'):
    """Loads train, validation, test dataset from SST raw data."""

    def get_label_value(label):
        return int(label)
    #     return label
    #     return {'0': 0.0, '1': 0.25, '2': 0.5, '3': 0.75, '4': 1.0}[label]

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

    _analyze_data(train_examples)

    train_data = torchtext.data.Dataset(train_examples, fields)
    valid_data = torchtext.data.Dataset(valid_examples, fields)
    test_data = torchtext.data.Dataset(test_examples, fields)

    text_field.build_vocab(train_data, vectors=glove)

    return train_data, valid_data, test_data, text_field.vocab

def create_iter(dataset, batch_size, device='cpu'):
    """
    Creates BucketIterator for given SST dataset.
    Set device='cuda' to use GPU.
    """
    return torchtext.data.BucketIterator(
        dataset, 
        batch_size=batch_size,
        device=device,
        sort_key=lambda x: len(x.text), # to minimize padding
        sort_within_batch=True,        # sort within each batch
        repeat=False)


def sst_analysis():
    train_set, valid_set, test_set, vocab = load_sst_dataset()
    print('Sample of vocab:', vocab.itos[:10])
    train_iter = create_iter(train_set, batch_size=32)
    for batch in train_iter:
        print(batch.label)
        data, lengths = batch.text
        # print(data)
        break

def manual_embeddings():
    test = split_text('the quick brown fox jumped over the green turtle. it was really exciting! HYPE?')
    embedding = convert_words_to_embeddings(test)
    print(embedding.shape, len(test))

if __name__ == '__main__':
    sst_analysis()

    
