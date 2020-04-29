# -*- coding: utf-8 -*-
import cv2
import os
import sys
import digit_detector.region_proposal as rp

import digit_detector.detect as detector
import digit_detector.file_io as file_io
import digit_detector.preprocess as preproc
import digit_detector.classify as cls
import ImgProcess.DetectDigit as ImgProcess

detect_model = "D:/Caps1/BGrade/Oh/capstone_project_bgrade_server - Copy/example/CAPS2/detector_model.hdf5"
recognize_model = "D:/Caps1/BGrade/Oh/capstone_project_bgrade_server - Copy/example/CAPS2/recognize_model.hdf5"

mean_value_for_detector = 200
mean_value_for_recognizer = 200

model_input_shape = (32, 32, 1)
DIR = os.getcwd() + '\\CAPS2\\data'


if __name__ == "__main__":
    # 1. image files
    print(DIR)
    print(os.getcwd())
    preproc_for_detector = preproc.GrayImgPreprocessor(mean_value_for_detector)
    preproc_for_recognizer = preproc.GrayImgPreprocessor(mean_value_for_recognizer)

    char_detector = cls.CnnClassifier(detect_model, preproc_for_detector, model_input_shape)
    char_recognizer = cls.CnnClassifier(recognize_model, preproc_for_recognizer, model_input_shape)

    digit_spotter = detector.DigitSpotter(char_detector, char_recognizer, rp.MserRegionProposer())
    path = sys.argv[1]
    index_student = ImgProcess.run(path)
    img_files = file_io.list_files(directory=DIR, pattern="*.jpg", recursive_option=False, n_files_to_sample=None,
                                   random_order=False)
    print(len(img_files))
    i = 0
    for img_file in img_files[0:]:
        i += 1
        # 2. image
        img = cv2.imread(img_file)

        res = digit_spotter.run(img, threshold=0.5, do_nms=True, nms_threshold=0.2)
        print(i, index_student[i-1], res[2])
