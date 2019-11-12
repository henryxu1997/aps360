import torch
import torchtext

from data_processing import split_text

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
    
    def get_sentiment(self, text):
        words = split_text(text)
        pos_count = 0
        for word in words:
            if word in self.positive_words:
                # print('pos', word)
                pos_count += 1
            elif word in self.negative_words:
                # print('neg', word)
                pos_count -= 1
        # If there's a balance of positive and negative words, return 0.5
        if pos_count == 0:
            return 0.5
        # Else return positive or negative depending on which has more.
        return 1.0 if pos_count > 0 else 0.0

def ucl_performance(model):
    count = 0
    total = 0
    path = "UCL_data/full_comments_labelled.txt"
    for line in open(path):
        text,label = line.split('\t')
        label = float(label)
        out = model.get_sentiment(text)
        if out == label:
            count+=1
        if out != 0.5:
            total+=1
    print(count,total)
    print(float(count)/total)

if __name__ == '__main__':
    print("baseline")
    model = BaselineModel()
    # x = model.get_sentiment('This cake is good.')
    # y = model.get_sentiment('The big fox is nasty')
    # print(x, y)
    ucl_performance(model)