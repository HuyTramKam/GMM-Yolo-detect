import cv2
import numpy as np

class GMMDetector:
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=35, detectShadows=True)

    def detect(self, frame):
        # 1. Làm mờ nhẹ để khử nhiễu hạt
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        # 2. Trừ nền
        fg_mask = self.backSub.apply(blurred)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # 4. Morphology tối giản (Chỉ làm sạch, không làm biến dạng)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # MORPH_OPEN: Xóa nhiễu li ti bên ngoài
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        # MORPH_CLOSE: Lấp lỗ hổng nhỏ bên trong cơ thể (giúp tâm ổn định)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400: 
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append((x, y, w, h))
                
        return boxes, fg_mask