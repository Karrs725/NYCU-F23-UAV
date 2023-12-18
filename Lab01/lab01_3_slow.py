import cv2
import numpy as np


img: np.ndarray = cv2.imread("./test.jpg")


div = 3
h, w, c = img.shape
upscaled_img = np.zeros((h * div, w * div, 3), np.uint8)

for iy, ix in np.ndindex(((h - 1) * div, (w - 1) * div)):
  ox, oy = ix // div, iy // div
  Q11, Q12 = img[oy, ox], img[oy + 1, ox]
  Q21, Q22 = img[oy, ox + 1], img[oy + 1, ox + 1]
  x1, y1 = ox * div, oy * div
  x2, y2 = x1 + div, y1 + div
  fxy1 = (x2 - ix) / div * Q11 + (ix - x1) / div * Q21
  fxy2 = (x2 - ix) / div * Q21 + (ix - x1) / div * Q22
  fxy = (y2 - iy) / div * fxy1 + (iy - y1) / div * fxy2
  upscaled_img[iy, ix] = fxy


cv2.imshow("upscaled_img", upscaled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import os
fn = os.path.basename(__file__).split(".")[0]
cv2.imwrite(f"./results/{fn}.jpg", upscaled_img)
