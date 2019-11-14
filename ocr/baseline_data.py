import csv
import os
import shutil
import sys

def copy_files(files, dest_dir):
    for f in files:
        dest = os.path.join(dest_dir, os.path.basename(f))
        shutil.copy(f, dest)

def separate_data(csv_file, data_dir):
    """
    Separate data into training and validation sets for use with baseline OCR.
    """
    word_count = {}

    with open(csv_file) as f:
        cr = csv.reader(f)
        for row in cr:
            img = row[0]
            label = row[1]
            if label not in word_count:
                word_count[label] = 0
            word_count[label] += 1

    sorted_words = sorted(
        word_count.items(), key=lambda kv: kv[1], reverse=True)
    sorted_words = sorted_words[:5] # top 5 most common words
    print(sorted_words)

    cwd = os.getcwd()

    num_train = 0
    num_valid = 0

    for word, _ in sorted_words:
        train_path = os.path.join(cwd, 'baseline_data', 'train', word)
        valid_path = os.path.join(cwd, 'baseline_data', 'valid', word)

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(valid_path, exist_ok=True)

        files = []
        with open(csv_file) as f:
            cr = csv.reader(f)
            for row in cr:
                img = row[0]
                label = row[1]
                if label == word:
                    img = os.path.join(data_dir, img)
                    files.append(img)
        
        split_idx = int(len(files) * 0.7)
        copy_files(files[:split_idx], train_path)
        copy_files(files[split_idx:], valid_path)

        num_train += split_idx
        num_valid += len(files) - split_idx
    
    print('Training samples:', num_train)
    print('Validation samples:', num_valid)

if __name__ == '__main__':
    csv_file = sys.argv[1]
    data_dir = sys.argv[2]

    separate_data(csv_file, data_dir)