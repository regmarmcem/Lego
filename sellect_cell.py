import cv2
import numpy as np
import copy
import random
import sys

if __name__ == "__main__":
    src = cv2.imread("empty.jpg", cv2.IMREAD_COLOR)
    if src is None:
        print("Failed to load image file.")
        sys.exit(1)

    height, width, channels = src.shape[:3]
    dst = copy.copy(src)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, binl = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    nLabels, labelImage = cv2.connectedComponents(binl)

    colors = []
    for i in range(1, nLabels + 1):
        colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

    for y in range(0, height):
        for x in range(0, width):
            if labelImage[y, x] > 0:
                dst[y, x] = colors[labelImage[y, x]]
            else:
                dst[y, x] = [0, 0, 0]

    cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Source", src)
    cv2.namedWindow("Connected Components", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Connected Components", dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
