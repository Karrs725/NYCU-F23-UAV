import cv2
import numpy as np


img: np.ndarray = cv2.imread("./test.jpg")


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)

kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

G_x = cv2.filter2D(img, -1, kernel).astype(np.float32)
G_y = cv2.filter2D(img, -1, kernel.T).astype(np.float32)

img = np.clip(np.sqrt(G_x ** 2 + G_y ** 2), 0, 255).astype(np.uint8)


cv2.imshow("Filtering & Sobel Operator", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import os
fn = os.path.basename(__file__).split(".")[0]
cv2.imwrite(f"./results/{fn}.jpg", img)
