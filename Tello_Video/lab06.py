import cv2
import numpy as np
import time
import math

from djitellopy import Tello
from keyboard_djitellopy import keyboard
from pyimagesearch.pid import PID


MAX_SPEED = 30
MIN_SPEED = -30


def detect_markers(frame, intrinsic, distortion, dictionary, parameters):
    markerCorners, markerIds, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion)
    if rvec is not None or tvec is not None: 
        for i in range(len(rvec)):
            frame = cv2.aruco.drawAxis(frame, intrinsic, distortion, rvec[i], tvec[i], 10)
            textCoordinate = ((int(markerCorners[i][0][0][0]) + int(markerCorners[i][0][2][0])) // 2 - 100, (int(markerCorners[i][0][0][1]) + int(markerCorners[i][0][2][1])) // 2)
            if textCoordinate[0] < 0: textCoordinate = (0, textCoordinate[1])
            if textCoordinate[0] > 400: textCoordinate = (400, textCoordinate[1])
            cv2.putText(frame, f"x: {tvec[0, 0, 0]:.2f}, y: {tvec[0, 0, 1]:.2f}, z: {tvec[0, 0, 2]:.2f}", textCoordinate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    return frame, rvec, tvec, markerIds


def main():
    drone = Tello()
    drone.connect()

    fs = cv2.FileStorage("uav_calibration.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()

    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters_create()

    x_pid = PID(0.7, 1e-4, 0.1)
    y_pid = PID(0.7, 1e-4, 0.1)
    z_pid = PID(0.7, 1e-4, 0.1)
    yaw_pid = PID(0.7, 1e-4, 0.1)

    x_pid.initialize()
    y_pid.initialize()
    z_pid.initialize()
    yaw_pid.initialize()

    drone.streamon()
    frame_read = drone.get_frame_read()
    drone.takeoff()
    
    while drone.is_flying:
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame, rvec, tvec, marker_ids = detect_markers(frame, intrinsic, distortion, dictionary, parameters)
        if rvec is None or tvec is None or len(marker_ids) == 0:
            drone.send_rc_control(0, 0, 0, 0)
            continue

        res = cv2.Rodrigues(rvec[0])[0]
        res = res @ np.array([0, 0, 1])
        x_update = tvec[0, 0, 0]
        x_update = x_pid.update(x_update, sleep=0)
        x_update = max(min(x_update, MAX_SPEED), MIN_SPEED)
        y_update = -(tvec[0, 0, 1] - 0)
        y_update = y_pid.update(y_update, sleep=0)
        y_update = max(min(y_update, MAX_SPEED), MIN_SPEED)
        z_update = tvec[0, 0, 2] - 100
        z_update = z_pid.update(z_update, sleep=0)
        z_update = max(min(z_update, MAX_SPEED), MIN_SPEED)
        yaw_update = math.atan2(res[0], res[2]) * 5
        yaw_update = yaw_pid.update(yaw_update, sleep=0)
        yaw_update = max(min(yaw_update, MAX_SPEED), MIN_SPEED)
        drone.send_rc_control(int(x_update), int(z_update), int(y_update * 1.5), int(-yaw_update))
        
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key != -1:
            keyboard(drone, key)

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
