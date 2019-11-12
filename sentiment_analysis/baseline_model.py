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
        self.record = {}
    
    def get_sentiment(self, text, split = True, return_pos_count = False):
        words = text
        if split:
            words = split_text(text)
        pos_count = 0
        for word in words:
            if word in self.positive_words:
                # print('pos', word)
                pos_count += 1
            elif word in self.negative_words:
                # print('neg', word)
                pos_count -= 1
        if pos_count not in self.record:
            self.record[pos_count] = 1
        else:
            self.record[pos_count] += 1
        if return_pos_count == True:
            return pos_count
        # If there's a balance of positive and negative words, return 0.5
        if pos_count == 0:
            return 0.5
        # Else return positive or negative depending on which has more.
        return 1.0 if pos_count > 0 else 0.0

    def print_record(self):
        print("start record")
        for record in sorted(self.record):
            print(record, self.record[record])
        print("end record")

    def reset_record(self):
        self.record = {}

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
    model.print_record()
    model.reset_record()
    print(count,total)
    print(float(count)/total)

def sst_performance(model):
    count = 0
    total = 0
    _,_,test_example = load_sst_dataset(return_example = True)
    for line in test_example:
        pos_count = model.get_sentiment(line.text,split = False,return_pos_count = True)
        out = -1
        if pos_count > 2:
            out = 4
        elif pos_count > 0:
            out = 3
        elif pos_count == 0:
            out = 2
        elif pos_count >= -2:
            out = 1
        else:
            out = 0

        if out == line.label:
            count+= 1
        total += 1
    model.print_record()
    model.reset_record()
    print(count,total)
    print(float(count)/total)


if __name__ == '__main__':
    print("baseline")
    model = BaselineModel()
    # x = model.get_sentiment('This cake is good.')
    # y = model.get_sentiment('The big fox is nasty')
    # print(x, y)
    sst_performance(model)