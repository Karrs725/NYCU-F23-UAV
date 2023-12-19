import cv2
import numpy as np

CAP = cv2.VideoCapture(0, cv2.CAP_DSHOW)
OBJ3D = np.zeros((54, 3), np.float32)
OBJ3D[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

corners = []
obj_points = []
img_points = []

while len(img_points) < 15:
    ret, frame = CAP.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret2, corner = cv2.findChessboardCorners(gray_frame, (9, 6), None)
    if ret2:
        key = cv2.waitKey(33)
        corners = cv2.cornerSubPix(gray_frame, corner, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        if key == 13:
            obj_points.append(OBJ3D)
            img_points.append(corners)
            print("append succeed: ", len(img_points))
        cv2.drawChessboardCorners(frame, (9, 6), corners, ret)

    cv2.imshow('frame', frame)
    cv2.waitKey(33)

ret, intrinsic, distortion, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (frame.shape[1], frame.shape[0]), None, None)

f = cv2.FileStorage("./notebook_calibration.xml", cv2.FILE_STORAGE_WRITE)
f.write("intrinsic", intrinsic)
f.write("distortion", distortion)
f.release()


cv2.waitKey(0)
cv2.destroyAllWindows()
