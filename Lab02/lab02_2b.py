import cv2
import numpy as np

img = cv2.imread('histogram.jpg')

img_HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

r = np.zeros(256)
for i in range(img_HSV.shape[0]):
    for j in range(img_HSV.shape[1]):
        r[img_HSV[i][j][2]]+=1

s = np.zeros(256)
s[0] = r[0]
for i in range(1,256):
    s[i] = r[i] + s[i-1]
s = np.round(s*256/(img.shape[0]*img.shape[1])).astype(int)

new_HSV = img_HSV.copy()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        new_HSV[i][j][2]=s[img_HSV[i][j][2]]
new_img = cv2.cvtColor(new_HSV,cv2.COLOR_HSV2BGR)
cv2.imshow('new_img',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('./results/lab02_2b.jpg', new_img)