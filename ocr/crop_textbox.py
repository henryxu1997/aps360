from PIL import Image
import sys
import argparse
import os

# parser = argparse.ArgumentParser(description='Crop a single textbox')
# parser.add_argument('--source_dir', default='../data/result', type=str, help='pretrained model')
# parser.add_argument('--target_dir', default='../data/singles', type=float, help='text confidence threshold')

# args = parser.parse_args()
# image_list, _, _ = file_utils.get_files(args.source_dir)
# target_folder = args.target_folder + "/"

# if not result_folder:
#     result_folder = './result/'
#     if not os.path.isdir(result_folder):
#         os.mkdir(result_folder)
# else:
#     if not os.path.isdir(args.result_folder):
#         print("Specified result folder does not exit.")
#         print("Created a folder:", Path(args.result_folder))
#         os.mkdir(result_folder)

# for k, image_path in enumerate(image_list):
#     print("Cropping Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
#     image = Image.open(image_path)

# if __name__ == '__main__':
def resizeImage(img):
    w, h = img.size
    w = int(round(w / 16)) * 16
    h = int(round(h / 16)) * 16
    return (w, h)

def cropImage(imgPath, coordPath, dirname, resizeImage=False):
    print(f'Cropping image {imgPath} {coordPath} {dirname}')
    img = Image.open(imgPath)
    root = os.path.splitext(os.path.basename(imgPath))[0]
    coords = open(coordPath, 'r').readlines()

    for i, coord in enumerate(coords):
        c = list(map(int,coord.split(',')))
        x1, y1, x2, y2 = c[0], c[1], c[4], c[5]
        img_cropped = img.crop((x1, y1, x2, y2))
        if resizeImage:
            print ("RESIZING")
            img_cropped = img.resize(resizeImage(img_cropped))
        img_cropped = img_cropped.convert("RGB")
        path = os.path.join(dirname, f'{root}_cropped_{str(i)}.jpg')
        img_cropped.save(path)
        print(f'Saved cropped file to: {path}')

if __name__ == '__main__':
    print('Intentionally empty. Can uncomment below to run function on an image PDF and a text file of coordinates.')
    # cropImage('../data/test/test1.png', '../data/result/res_test1.txt', './data')
