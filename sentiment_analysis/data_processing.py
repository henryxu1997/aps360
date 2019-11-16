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

'''
def convert_words_to_embeddings(words):
    """
    words = List[str]
    """
    x = torch.zeros(size=[len(words), EMBEDDING_SIZE])
    for i, word in enumerate(words):
        x[i] = glove[word]
    return x
'''

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


def load_sst_dataset(root='.data', analyze_data=False, char_base = False, three_labels = False, regression = False):
    """Loads train, validation, test dataset from SST raw data."""

    def get_label_value(label):
        return int(label)
    def get_three_label(label):
        int_label = int(label)
        if int_label <2:
            return 0
        if int_label >2:
            return 2
        return 1
    def split_char(x):
        return x

    train_token = split_char if char_base else split_text
    label_type = torch.float if regression else torch.long
    label_func = get_three_label if three_labels else get_label_value

    # Define text and label fields
    text_field = torchtext.data.Field(
        sequential=True,tokenize = train_token, batch_first=True, include_lengths=True)
    label_field = torchtext.data.Field(
        sequential=False, batch_first=True, use_vocab=False,
        dtype=label_type,
        preprocessing=torchtext.data.Pipeline(label_func))

    fields = [('text', text_field), ('label', label_field)]

    # This will download SST dataset to .data if not already present.
    path = torchtext.datasets.SST.download(root)

    # Train, validation, and test data already in separate files.
    train_path = os.path.join(path, 'train.txt')
    valid_path = os.path.join(path, 'dev.txt')
    test_path = os.path.join(path, 'test.txt')

    with open(os.path.expanduser(train_path)) as f:
        train_examples = [torchtext.data.Example.fromtree(line, fields) for line in f]
    with open(os.path.expanduser(valid_path)) as f:
        valid_examples = [torchtext.data.Example.fromtree(line, fields) for line in f]
    with open(os.path.expanduser(test_path)) as f:
        test_examples = [torchtext.data.Example.fromtree(line, fields) for line in f]

    if analyze_data:
        _analyze_data(train_examples)

    train_data = torchtext.data.Dataset(train_examples, fields)
    valid_data = torchtext.data.Dataset(valid_examples, fields)
    test_data = torchtext.data.Dataset(test_examples, fields)


    if char_base:
        text_field.build_vocab(train_data)
    else:
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
    print('Train size:', len(train_set))
    print('Valid size:', len(valid_set))
    print('Test size:', len(test_set))
    print('Training sample:', train_set[0].text, train_set[0].label)
    print('Sample of vocab:', vocab.itos[:10])

    train_iter = create_iter(train_set, batch_size=32)
    for batch in train_iter:
        print(batch.label)
        data, lengths = batch.text
        # print(data)
        break

'''
def manual_embeddings():
    test = split_text('the quick brown fox jumped over the green turtle. it was really exciting! HYPE?')
    embedding = convert_words_to_embeddings(test)
    print(embedding.shape, len(test))
'''

def glove_exploration():
    words = 'happy sad king queen relevant hyperbole'.split()
    for word in words:
        print(glove[word])
    print(dir(glove))
    print(len(glove.itos))

if __name__ == '__main__':
    sst_analysis()
    glove_exploration()

    
