import numpy as np
import sys
import os
import cv2


if __name__ == "__main__":
    
 #   print(sys.argv[1])
 #   img = cv2.imread(sys.argv[1], 0)
 #   cv2.imshow('img', img)
 #   cv2.waitKey(0)
    img = np.array([[1, 2], [1, 3], [1, 4]])
    print(img.tolist())
    sys.stdout.flush()
