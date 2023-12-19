import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID


OBJ3D = np.zeros((54, 3), np.float32)
OBJ3D[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

corners = []
obj_points = []
img_points = []

def main():
    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()

    while len(img_points) < 15:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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

    f = cv2.FileStorage("./uav_calibration.xml", cv2.FILE_STORAGE_WRITE)
    f.write("intrinsic", intrinsic)
    f.write("distortion", distortion)
    f.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



