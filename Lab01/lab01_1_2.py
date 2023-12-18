import cv2
import numpy as np


CONTRAST = 100
BRIGHTNESS = 40


img: np.ndarray = cv2.imread("./nctu_flag.jpg")
high_contrast_img = img.copy()


filter = (img[..., 0] + img[..., 1]) * 0.3 <= img[..., 2]

high_contrast_img = np.clip((img.astype(np.int32) - 127) * (CONTRAST / 127 + 1) + 127 + BRIGHTNESS, 0, 255).astype(np.uint8)
high_contrast_img[filter] = img[filter]


cv2.imshow("high_contrast_img", high_contrast_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import os
fn = os.path.basename(__file__).split(".")[0]
cv2.imwrite(f"./results/{fn}.jpg", high_contrast_img)
