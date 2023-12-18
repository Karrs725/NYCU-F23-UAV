import cv2
import numpy as np


img: np.ndarray = cv2.imread("./test.jpg")


div = 3
h, w, c = img.shape
upscaled_img = np.zeros((h * div, w * div, 3), np.uint8)


for iy, ix in np.ndindex((h, w)):
    upscaled_img[iy * div:(iy + 1) * div, ix * div:(ix + 1) * div] = img[iy, ix]


cv2.imshow("upscaled_img", upscaled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import os
fn = os.path.basename(__file__).split(".")[0]
cv2.imwrite(f"./results/{fn}.jpg", upscaled_img)
