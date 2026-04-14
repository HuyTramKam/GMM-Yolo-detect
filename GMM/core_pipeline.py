import cv2


class CorePipeline:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def preprocess(self, frame):
        frame = cv2.resize(frame, (640, 480))
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
