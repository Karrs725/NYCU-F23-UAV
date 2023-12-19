import cv2
import numpy as np
import time
import math
from djitellopy import Tello
from pyimagesearch.pid import PID


def main():
    # Tello
    drone = Tello()
    drone.connect()
    #time.sleep(10)
    drone.streamon()
    frame_read = drone.get_frame_read()
    
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()
    while True:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        
        fs = cv2.FileStorage("./uav_calibration.xml", cv2.FILE_STORAGE_READ)
        intrinsic = fs.getNode("intrinsic").mat()
        distortion = fs.getNode("distortion").mat()
        
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
        if rvec is not None or tvec is not None: 
            for i in range(len(rvec)):
                frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec[i], tvec[i], 10)
                textCoordinate = ((int(markerCorners[i][0][0][0]) + int(markerCorners[i][0][2][0])) // 2 - 100, (int(markerCorners[i][0][0][1]) + int(markerCorners[i][0][2][1])) // 2)
                if textCoordinate[0] < 0: textCoordinate = (0, textCoordinate[1])
                if textCoordinate[0] > 400: textCoordinate = (400, textCoordinate[1])
                cv2.putText(frame, f"x: {tvec[0, 0, 0]:.2f}, y: {tvec[0, 0, 1]:.2f}, z: {tvec[0, 0, 2]:.2f}", textCoordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(33)
        if key == 27: break
    
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

