from PIL import Image
import pytesseract
import os

p = os.getcwd()


def pdfparser():
    img = Image.open('D:/Caps2/Re_Digit/CAPS2/ImgProcess/table.jpg', 'r')
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(img, lang='eng')
    file = open(p + '\\CAPS2\\ImgProcess\\textTA.txt', "wb")
    file.write(text.encode('utf-8'))
    file.close()
    return read()


def read():
    file = open(p + '\\CAPS2\\ImgProcess\\textTA.txt', 'r', encoding="utf8")
    result = []
    for line in file:
        line = line.strip().replace('|', ' ')
        if len(line) == 0:
            continue
        else:
            l = line.split(' ')
            for it in l:
                if it.isdigit() and int(it) > 100000000:
                    result.append(it)
    return result

