killall preview

demo_file_path="$1"
open $demo_file_path
export GOOGLE_APPLICATION_CREDENTIALS=./api.json

# Reset all directories
rm -rf data/demo/*
mkdir -p data/demo/raw_file
mkdir -p data/demo/characters
mkdir -p data/demo/text
mkdir -p data/demo/voice

cp $demo_file_path data/demo/raw_file/$(basename $demo_file_path)
sleep 2s
# #1. Character Cropping
echo "Cropping characters from $demo_file_path ..."
python3 character_segmentation-master/main.py --output_letter_size 128 --input_folder data/demo/raw_file/ --output_folder data/demo/characters/
open Source_Image.jpg
sleep 8s
open ./data/demo/characters
sleep 5s
killall preview
killall finder

# #2. Character Recognition
echo "Reading characters from data/demo/characters/ ..."
python3 character_recognition/train.py --input_folder data/demo/characters/ --output_folder data/demo/text/ --model='character_recognition/models/nc=62:F=3:M=5:lr=0.01:epoch=010.pt'
open ./data/demo/text/
sleep 5s
open ./output.txt
sleep 16s
#3. Sentimental Analysis + Speech Generation + Play Speech
echo "Generating sentiment + Speech ..."
python3 sentiment_analysis/sentiment_analysis.py ./output.txt data/demo/voice/output
