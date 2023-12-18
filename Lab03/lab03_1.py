import cv2
import numpy as np


img: np.ndarray = cv2.imread("./test.jpg")
iter_num = 15


rect = cv2.selectROI("select a rectangle in test image", img)
cv2.destroyAllWindows()


b_Model = np.zeros((1, 65), np.float64)
f_Model = np.zeros((1, 65), np.float64)


mask_new, b_model, f_model=cv2.grabCut(img, None, rect, b_Model, f_Model, iter_num, 
cv2.GC_INIT_WITH_RECT)

mask = np.where((mask_new == 0) | (mask_new == 2), 0, 1).astype(np.uint8)
img = img * mask[:, :, np.newaxis]


cv2.imshow("GrabCut Algorithm", img)
cv2.waitKey(0)
cv2.destroyAllWindows()