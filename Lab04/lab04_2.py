import cv2
import numpy as np


CAP = cv2.VideoCapture(0, cv2.CAP_DSHOW)
img = cv2.imread("screen.jpg")

def bilinear_interpolation(image, x, y):
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1
    if x1 >= image.shape[1] or y1 >= image.shape[0]: return image[y0, x0]
    dx, dy = x - x0, y - y0
    return (1 - dx) * (1 - dy) * image[y0, x0] + dx * (1 - dy) * image[y0, x1] + (1 - dx) * dy * image[y1, x0] + dx * dy * image[y1, x1]


points = [
    [276.43, 188.43],
    [617.45, 88.39],
    [264.37, 399.38],
    [626.39, 368.37]
]


while True:
    ret, frame = CAP.read()
    transform_matrix = cv2.getPerspectiveTransform(np.float32(points), np.float32([[0, 0], [frame.shape[1], 0], [0, frame.shape[0]], [frame.shape[1], frame.shape[0]]]))
    transformed_frame = img
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            ox, oy, scale = transform_matrix @ np.asarray([x, y, 1])
            ox /= scale
            oy /= scale
            if ox < 0 or ox >= frame.shape[1] or oy < 0 or oy >= frame.shape[0]: continue
            transformed_frame[y, x] = bilinear_interpolation(frame, ox, oy)
    cv2.imshow("frame", transformed_frame)
    key = cv2.waitKey(10)
    if key == 27: break