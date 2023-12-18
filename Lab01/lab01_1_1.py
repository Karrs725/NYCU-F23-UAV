import cv2
import numpy as np


img: np.ndarray = cv2.imread("./nctu_flag.jpg")
filtered_img = img.copy()


filter = (img[..., 0] > 100) & (img[..., 0] * 0.8 > img[..., 1]) & (img[..., 0] * 0.8 > img[..., 2])

filtered_img[..., :] = np.expand_dims(filtered_img.sum(-1) / 3, -1).repeat(3, -1)
filtered_img[filter] = img[filter]


cv2.imshow("filtered_img", filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import os
fn = os.path.basename(__file__).split(".")[0]
cv2.imwrite(f"./results/{fn}.jpg", filtered_img)
