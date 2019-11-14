import torch
import torchtext

from data_processing import split_text, load_sst_dataset

class BaselineModel:
    """
    Baseline model stores a list of positive and negative words.
    When analyzing a new text, it counts the number of positive and negative words
    in that text and uses that to determine the overall sentiment of the sentence.
    """
    def __init__(self):
        with open('data/positive_words.txt') as f:
            self.positive_words = set([s.strip() for s in f.readlines()])
        with open('data/negative_words.txt') as f:
            self.negative_words = set([s.strip() for s in f.readlines()])
    
    def get_sentiment(self, words):
        pos_count = 0
        for word in words:
            if word in self.positive_words:
                # print('pos', word)
                pos_count += 1
            elif word in self.negative_words:
                # print('neg', word)
                pos_count -= 1
        if pos_count >= 2:
            return 4
        elif pos_count == 1:
            return 3
        elif pos_count == 0:
            return 2
        elif pos_count == -1:
            return 1
        else:
            return 0

def ucl_performance(model):
    correct, total = 0, 0
    path = "data/ucl/full_comments_labelled.txt"
    for line in open(path):
        text, label = line.split('\t')
        label = int(label)
        words = split_text(text)
        sentiment = model.get_sentiment(words)
        # print(words, label, sentiment)

        # Convert 0 to either (0,1) negative sentiment
        # Convert 1 to either (3,4) positive sentiment
        if label == 0 and sentiment in [0,1]:
            correct += 1
        elif label == 1 and sentiment in [3,4]:
            correct += 1
        total += 1
    acc = correct / total
    return acc

def sst_performance(model, verbose=False):
    correct, total = 0, 0
    _, __, test_set, ___ = load_sst_dataset()
    wrong = []
    for example in test_set:
        example_txt = ' '.join(example.text)
        tokens = split_text(example_txt)
        sentiment = model.get_sentiment(tokens)
        if sentiment == example.label:
            correct += 1
        else:
            wrong.append((example_txt, example.label, sentiment))
        total += 1
    if verbose:
        print(f'Number of wrong {len(wrong)}')
        print(wrong)
    acc = correct / total
    return acc


if __name__ == '__main__':
    model = BaselineModel()
    print(f'UCL accuracy: {ucl_performance(model)}')
    print(f'SST accuracy: {sst_performance(model)}')
    x = model.get_sentiment(split_text('This cake is good.'))
    y = model.get_sentiment(split_text('The big fox is nasty'))
    print(x, y)