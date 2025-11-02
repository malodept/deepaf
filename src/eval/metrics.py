
import cv2 as cv
import numpy as np

def tenengrad(img_gray):
    gx = cv.Sobel(img_gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(img_gray, cv.CV_32F, 0, 1, ksize=3)
    g2 = gx*gx + gy*gy
    return float(g2.mean())

def laplacian_sharpness(img_gray):
    lap = cv.Laplacian(img_gray, cv.CV_32F, ksize=3)
    return float(lap.var())
