from text_to_sentiment import get_sentiment
from text_to_speech import synthesize_speech
from pydub import AudioSegment
from pydub.playback import play
import sys

def main():
    if sys.argv[1]:
        text_list = open(sys.argv[1], 'r').readlines()
        sentiment_list = get_sentiment(text_list, "./sentiment_analysis/")
        synthesize_speech(text_list, sentiment_list, outputPath=sys.argv[2])
        play(AudioSegment.from_mp3(sys.argv[2]+".mp3"))
    else:
        print("No textfile found.")

main()