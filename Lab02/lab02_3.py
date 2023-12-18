import cv2
import numpy as np


img: np.ndarray = cv2.imread("./otsu.jpg")


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
histogram, indices = np.histogram(img, range(256))


sigmas = []

n_B = 0
n_O = histogram.sum()
m_B = 0
m_O = img.mean()


for T in range(0, 255 - 1):
    sigmas.append(n_B * n_O * (m_B - m_O) ** 2)
    n = histogram[T]
    new_n_B = n_B + n
    new_n_O = n_O - n
    m_B = (m_B * n_B + n * T) / new_n_B
    m_O = (m_O * n_O - n * T) / new_n_O
    n_B = new_n_B
    n_O = new_n_O


T = np.argmax(sigmas)
img[img >= T] = 255
img[img < T] = 0


cv2.imshow("Otsu Threshold", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import os
fn = os.path.basename(__file__).split(".")[0]
cv2.imwrite(f"./results/{fn}.jpg", img)
