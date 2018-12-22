import pytesseract

def ocr(image):
    return pytesseract.image_to_data(image, lang='jpn+jpn_vert', output_type=pytesseract.Output.DICT)
