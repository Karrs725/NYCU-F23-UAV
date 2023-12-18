import cv2
import numpy as np

img = cv2.imread('histogram.jpg')

r = np.zeros((256,3))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(3):
            r[img[i][j][k]][k]+=1

s = np.zeros((256,3))
s[0] = r[0]
for i in range(1,256):
    s[i] = r[i] + s[i-1]
s = np.round(s*256/(img.shape[0]*img.shape[1])).astype(int)

new_img=np.zeros(img.shape,np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(3):
            new_img[i][j][k]=s[img[i][j][k]][k]

cv2.imshow('new_img',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('./results/lab02_2a.jpg', new_img)