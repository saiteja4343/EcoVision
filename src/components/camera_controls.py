import cv2

def list_available_cameras():
    cameras = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras
