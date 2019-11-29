import os
import string

import torch
from torchvision import transforms, datasets
import cv2
import pytesseract
from PIL import Image
from PIL import ImageDraw

from train import load_model
from network import ALPHABET

# Load model globally
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
model_path = 'models/nc=62:F=3:M=5:lr=0.01:epoch=010.pt'
network = load_model(model_path)


FILE_PATH = './jordan_work/hp_pg16.jpg'

def image_char_extraction(file_path=FILE_PATH):
    target = pytesseract.image_to_string(file_path)
    print(target)
    bbox_info = pytesseract.image_to_boxes(file_path, output_type='string')
    target_i = 0

    output_text = []
    correct, total = 0, 0
    with Image.open(file_path).convert('RGB') as img:
        # Convert to monochrome
        # img = img.convert('L')
        print(img.size)
        width, height = img.size
        print(img.getbbox())

        for line in bbox_info.split('\n'):
            tokens = [t.strip() for t in line.split()]
            char = tokens[0]
            # Verify the bbox_info and target are aligned
            if char != target[target_i]:
                print(f'mismatch {char} {target[target_i]} {target_i}')
            '''
            else:
                print(char, end='')
            '''
            bbox = [int(i) for i in tokens[1:-1]]
            # print('OLD BBOX', bbox)
            # Enlarge the box a bit so it's not super tight around character
            bbox[0] -= 7
            bbox[2] += 7
            tmp = bbox[1]
            # Reverse the y coordinates
            bbox[1] = (height - bbox[3]) - 7
            bbox[3] = (height - tmp) + 7
            # print('NEW', bbox)

            # draw = ImageDraw.Draw(img)
            # draw.rectangle(bbox, fill='red')
            # img.show()
            
            # The Python Imaging Library uses a Cartesian pixel coordinate system, with 
            # (0,0) in the upper left corner. Note that the coordinates refer to the implied 
            # pixel corners; the centre of a pixel addressed as (0, 0) actually lies at (0.5, 0.5).
            cropped_img = img.crop(bbox)
            # old_size = cropped_img.size
            # new_size = (128, 128)
            # new_im = Image.new("RGB", new_size, color='white')   ## luckily, this is already black!
            # new_im.paste(cropped_img, [int(i) for i in ((new_size[0]-old_size[0])/2, (new_size[1]-old_size[1])/2)])

            # new_im.show()
            # return
            
            if char in ALPHABET:
                predicted_letter = run_img_predict_char(cropped_img)
                output_text.append(predicted_letter)
                if predicted_letter.lower() == char.lower():
                    correct += 1
                total += 1
            else:
                output_text.append(char)

            # Advance a character and skip any whitespace
            target_i += 1
            while target_i < len(target) and target[target_i] in string.whitespace:
                target_i += 1
                output_text.append(' ')
                # print(' ', end='')

    ans = ''.join(output_text)
    print(ans) 
    print('Accuracy', correct/total)

def get_letter(outputs):
    output_prob = torch.softmax(outputs, dim=1)
    # print(output_prob)
    _, indices = output_prob.max(1)
    return ALPHABET[indices]

def test_network_on_screenshot_letters(folder='jordan_work/pg16'):
    last_token = 0
    print('# character images = ', len(os.listdir(folder)))
    ans_text = open(os.path.join(folder, 'ans.txt')).read()
    ans_text = ''.join(ans_text.split())
    print(ans_text)
    i = 0
    correct, total = 0, 0
    for path in sorted(os.listdir(folder), key=lambda item: (len(item), item)):
        if not path.endswith('png'):
            continue
        try:
            tokens = path.split('-')
            if int(tokens[0]) != last_token:
                last_token = int(tokens[0])
                print(' ')
        except Exception as e:
            pass
        with Image.open(f'{folder}/{path}').convert('RGB') as img:
            predicted_char = run_img_predict_char(img)
            print(predicted_char, end='')
            if predicted_char.lower() == ans_text[i].lower():
                correct += 1
            i += 1
            total += 1

    print(' DONE')
    print(correct/total)

def run_img_predict_char(img, show_img=False):
    img = img.resize((128,128))
    if show_img:
        print(img.size)
        img.show()
    tensor_img = transform(img).unsqueeze(0)
    out = network(tensor_img)
    return get_letter(out)

if __name__ == '__main__':
    # image_char_extraction()
    test_network_on_screenshot_letters()