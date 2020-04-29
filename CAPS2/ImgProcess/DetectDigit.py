from pdf2image import convert_from_path
import cv2
import os
import ImgProcess.ExtractId as ExtractId


p = os.getcwd()

def resize(img, scale_per):
    width = int(img.shape[1] * scale_per / 100)
    height = int(img.shape[0] * scale_per / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def create_mask(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    horizal = thresh
    vertical = thresh

    scale_height = 30
    scale_long = 20

    long = int(img.shape[1] / scale_long)
    height = int(img.shape[0] / scale_height)

    horizalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (long, 1))
    horizal = cv2.erode(horizal, horizalStructure, (-1, -1))
    horizal = cv2.dilate(horizal, horizalStructure, (-1, -1))

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

    return vertical + horizal


def get_nghieng(img):
    mask = create_mask(img)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max = -1
    rect = []
    for cnt in contours:
        if cv2.contourArea(cnt) > max:
            max = cv2.contourArea(cnt)
            rect = cv2.minAreaRect(cnt)
    ng = rect[2]
    if ng < -45:
        ng = 90 + ng
    return ng


def fix_img(img):
    nghieng = get_nghieng(img)
    cols = img.shape[1]
    rows = img.shape[0]
    m5 = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=nghieng, scale=1)
    return cv2.warpAffine(img, m5, (cols, rows))


def get_table(img):

    def set_size(img, width, height):
        t = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        table = t[135:1995, :]
        return table

    img = fix_img(img)
    mask = create_mask(img)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max = -1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > max:
            x_max, y_max, w_max, h_max = x, y, w, h
            max = cv2.contourArea(cnt)

    table = img[y_max:y_max + h_max, x_max:x_max + w_max]
    return set_size(table, 2200, 2000)


def split(img_name, num):
    img = cv2.imread(img_name, 0)
    for i in range(num):
        cv2.imwrite(p + '\\CAPS2\\data\\img{}.jpg'.format(i), img[62 * i: 62 * (i + 1) - 2, :])


def run(path):
    pages = convert_from_path(path, 500)
    pages[0].save(p + '\\CAPS2\\ImgProcess\\out.jpg', 'JPEG')

    img = cv2.imread(p + '\\CAPS2\\ImgProcess\\out.jpg', 0)
    table = get_table(img)
    cv2.imwrite(p + '\\CAPS2\\ImgProcess\\table.jpg', table)
    k = table[:, 1658:1743]
    cv2.imwrite(p + '\\CAPS2\\ImgProcess\\digit.jpg', k)
    index_student = ExtractId.pdfparser()
    split(p + '\\CAPS2\\ImgProcess\\digit.jpg', len(index_student))
    return index_student

