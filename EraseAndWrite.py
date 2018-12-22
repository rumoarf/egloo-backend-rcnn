from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import pytesseract
import translate
import json


def write(image, data):
    font = ImageFont.truetype("D2Coding.ttc", 10)
    original = ''.join(data['text']).strip()
    if original:
        translated = json.loads(translate.translate(original))['message']['result']['translatedText']
        ImageDraw.Draw(image).multiline_text(text=translated, xy=(10, 10), font=font, fill=(255, 0, 255))
        
    return image


def erase(image, data, color='blue'):
    imageWidth, imageHeight = image.size
    for (char, left, top, width, height) in zip(data['text'], data['left'], data['top'], data['width'], data['height']):
        if char and width != imageWidth and height != imageHeight:
            ImageDraw.Draw(image).rectangle(xy=((left, top), (left + width, top + height)), outline=color, fill='white')
    return image

# test_image = Image.open('test3.png')
# original_text = pytesseract.image_to_string(test_image, lang='jpn+jpn_vert')
# print(original_text)
# translated_text = json.loads(translate.translate(original_text))['message']['result']['translatedText']
# print(translated_text)
# data = pytesseract.image_to_data(test_image, lang='jpn+jpn_vert', output_type=pytesseract.Output.DICT)

# erase(test_image, data)

# # write(test_image, 50, 36, translated_text)
# test_image.show()
# for line in pytesseract.image_to_boxes(test_image, lang='jpn+jpn_vert').split('\n'):
#     print(line)
# write(Image.open('test.png'), 10, 10, "test")
