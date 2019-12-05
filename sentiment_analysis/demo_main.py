import re
import sys

from text_to_sentiment import get_sentiment

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise ValueError('Must specify folder name')
    folder = sys.argv[1]
    with open(f'../character_recognition/demo_outputs/ocr_{folder}_output.txt') as f:
        sentences = re.split(r'[.!?]\s*', f.read().strip())[:-1]
        print(sentences)
        get_sentiment(sentences)