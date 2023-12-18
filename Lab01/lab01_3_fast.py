import cv2
import numpy as np


DIV = 3


img: np.ndarray = cv2.imread("./test.jpg")
h, w, c = img.shape


Q11 = np.roll(img, 1, axis=0)
Q12 = img
Q21 = np.roll(img, 1, axis=1)
Q22 = np.roll(Q11, 1, axis=1)

Q = np.stack((np.stack((Q11, Q12), axis=0), np.stack((Q21, Q22), axis=0)), axis=-1)

divisions = np.asarray([[i, DIV - i] for i in range(DIV)])

# print(divisions.shape, Q.shape, divisions.T.shape)

# current: (div_w, 2, 2, h, w, c, 2, 2, div_h)
upscaled_img = np.einsum("ij, jklmn, no -> iklmo", divisions, Q, divisions.T) / DIV ** 2

# current: (div_w, h, w, c, div_h)
upscaled_img = np.moveaxis(upscaled_img, 0, -2)
# current: (h, w, c, div_w, div_h)
upscaled_img = np.concatenate(upscaled_img, axis=-1)
# current: (w, c, div_w, h * div_h)
upscaled_img = np.moveaxis(upscaled_img, -1, 1)
# current: (w, h * div_h, c, div_w)
upscaled_img = np.concatenate(upscaled_img, axis=-1)
# current: (h * div_h, c, w * div_w)
upscaled_img = np.moveaxis(upscaled_img, -1, 1)
# current: (h * div_h, w * div_w, c)
upscaled_img = upscaled_img.astype(np.uint8)


cv2.imshow("upscaled_img", upscaled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


import os
fn = os.path.basename(__file__).split(".")[0]
cv2.imwrite(f"./results/{fn}.jpg", upscaled_img)
