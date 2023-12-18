import cv2
import numpy as np

T = 500

cap = cv2.VideoCapture('train.mp4')
backSub = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = backSub.apply(frame)
    shadowval = backSub.getShadowValue()
    ret, nmask = cv2.threshold(fgmask, shadowval, 255, cv2.THRESH_BINARY)

    num_labels, labels = cv2.connectedComponents(nmask, connectivity=4)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if 0 in unique_labels:
        unique_labels = unique_labels[1:]
        label_counts = label_counts[1:]
    
    for i in range(len(unique_labels)):
        if label_counts[i] <= T:
            continue
        indices = np.where(labels == unique_labels[i])
        U, D, L, R = indices[0].min(), indices[0].max(), indices[1].min(), indices[1].max()
        cv2.rectangle(frame, (R,U), (L,D), (0, 255, 0), thickness=2)


    cv2.imshow('frame',frame)
    cv2.waitKey(33)
    

cap.release()
cv2.destroyAllWindows()