import cv2
import numpy as np


CAP = cv2.VideoCapture(0, cv2.CAP_DSHOW)

FACE_COORDINATE = np.float32([0, 0, 15, 25]) # face coordinate for real world faces
PEOPLE_COORDINATE = np.float32([0, 0, 75, 175]) # people coordinate for real world people

def detect_people(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(frame, winStride=(8, 8), scale=1.05, useMeanshiftGrouping=False)
    #if len(rects) > 0: rects = rects[weights.ravel() > 0.5]
    return rects

def detect_frontal_face(frame):
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    rects = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.2, 5, cv2.CASCADE_SCALE_IMAGE, (32, 32))
    return rects

def draw_rects(frame, rects, color=(0, 255, 0)):
    for x, y, w, h in rects:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame

def mark_depth(frame, face_rect, real_rect, color):
    x1, y1, w1, h1 = real_rect
    x2, y2, w2, h2 = face_rect
    fs = cv2.FileStorage("notebook_calibration.xml", cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode("intrinsic").mat()
    distortion = fs.getNode("distortion").mat()
    fs.release()
    success, rvec, tvec = cv2.solvePnP(np.float32([[x1, y1, 0], [x1+w1, y1, 0], [x1+w1, y1+h1, 0], [x1, y1+h1, 0]]), np.float32([[x2, y2], [x2+w2, y2], [x2+w2, y2+h2], [x2, y2+h2]]), intrinsic, distortion)
    if success: frame = cv2.putText(frame, f"depth: {tvec[2, 0]:.2f} cm", (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def main():
    while True:
        _, frame = CAP.read()
        rects = detect_people(frame)
        frame = draw_rects(frame, rects, (0, 255, 255))
        for rect in rects: frame = mark_depth(frame, rect, PEOPLE_COORDINATE, (0, 255, 255))
        rects = detect_frontal_face(frame)
        frame = draw_rects(frame, rects, (0, 255, 0))
        for rect in rects: frame = mark_depth(frame, rect, FACE_COORDINATE, (0, 255, 0))
        cv2.imshow("frame", frame)
        key = cv2.waitKey(33)
        if key == 27: break


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
