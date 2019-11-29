import os

import torch
from data_processing import load_sst_dataset, split_text
from network import WordSANet

def get_sentiment(texts, modelPath):
    """
    Given list of strings, returns list of sentiments.
    negative = 0, neutral = 1, positive = 2
    """

    saved_model_file = modelPath + '/./saved_66.9p_epoch10/saved_models/WordSANet:16531:200:lstm:108:1:0.0:lr=0.0007:wd=0.0:b=64epoch=10.pt'

    _, _, _, vocab = load_sst_dataset(
        char_base=False, three_labels=True, regression=False)

    model = WordSANet(vocab.vectors, layer_type='lstm',
        output_size=3,regression=False, hidden_size=108,
        num_layers=1, dropout=0.0)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    sentiments = []
    for sentence in texts:
        sentence_words = split_text(sentence)
        if len(sentence_words) < 1:
            sentiments.append(1)
            continue
        word_tensor = torch.zeros(len(sentence_words), dtype=int)
        for i, word in enumerate(sentence_words):
            index = vocab.stoi[word]
            word_tensor[i] = index
        out = model(word_tensor.unsqueeze(0))
        print(out)
        output_prob = torch.softmax(out, dim=1)
        # indices from 0-2 indicating sentiment class
        _, indices = output_prob.max(1)
        sentiments.append(indices[0].item())

    return sentiments

if __name__ == '__main__':
    texts = [
        'Ivan is the coolest most awesome person in the world!', # positive
        'Ivan is okay, but not the greatest.', # neutral
        'Ivan is absolutely terrible and I abhor him.'] # negative
    print(get_sentiment(texts))
