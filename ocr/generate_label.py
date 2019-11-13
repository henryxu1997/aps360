try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import os
import csv
import sys
import pandas as pd

def generateLabelsForTrainingImages(data_path, csv_file="next_input.csv"):

	# If you don't have tesseract executable in your PATH, include the following with updated PATH to tesseract:
	# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

	file_names = []
	write_to_csv = [("Path", "Label")]
	valid_file_types = [".jpg", ".png"]

	# create label for each training image
	for f in os.listdir(data_path):
		if f.endswith(valid_file_types[0]) or f.endswith(valid_file_types[1]):
			label = pytesseract.image_to_string(Image.open(data_path+f))
			if label:
				write_to_csv.append((data_path + "/" + f, label))

	# write training image file name and label into CSV
	with open(csv_file, "w+") as csv_file:
		writer = csv.writer(csv_file, delimiter = ',')
		for line in write_to_csv:
			writer.writerow(line)
	
	return readPathsFromCsv(csv_file)

def readPathsFromCsv(csv_file="next_input.csv"):
	df = pd.read_csv(csv_file)
	imagePathList = df.Path
	labelList = df.Label
	return (imagePathList, labelList)


# argv[1] is path to training images
if __name__ == "__main__":
	data_path = sys.argv[1]
	generateLabelsForTrainingImages(data_path)



