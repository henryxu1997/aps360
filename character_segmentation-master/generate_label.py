try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import os
import csv
import sys
import errno


def generateLabelsForTrainingImages(data_path, output_path):

	possible_values = {}
	for index in range(26):
		possible_values[(chr(ord('a') + index))] = "_lower"

	for index in range(26):
		possible_values[(chr(ord('A') + index))] = "_upper"

	for index in range(10):
		possible_values[str(index)] = "_digit"

	# If you don't have tesseract executable in your PATH, include the following with updated PATH to tesseract:
	# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
	file_names = []
	write_to_csv = [("File name", "Label")]
	valid_file_types = [".jpg", ".png"]

	# create label for each training image
	for f in os.listdir(data_path):
		if f.endswith(valid_file_types[0]) or f.endswith(valid_file_types[1]):

			# psm 10 is for character recognition
			label = pytesseract.image_to_string(Image.open(data_path+f), config='--psm 10')
			
			# only save labels within the 62 that we are interested in
			if label and label in possible_values:
				if not os.path.exists(output_path + label + possible_values[label]):
					try:
						os.makedirs(output_path + label + possible_values[label])
					except OSError as e:
						if e.errno != errno.EEXIST:
							raise
				curr_image = Image.open(data_path+f)
				curr_image.save(output_path + label + possible_values[label] + '/' + f)
				write_to_csv.append((f, label))

	# write training image file name and label into CSV
	with open("next_input.csv", "w+") as csv_file:
		writer = csv.writer(csv_file, delimiter = ',')
		for line in write_to_csv:
			writer.writerow(line)

# argv[1] is path to training images
if __name__ == "__main__":
	data_path = sys.argv[1]
	output_path = sys.argv[2]
	generateLabelsForTrainingImages(data_path, output_path)



