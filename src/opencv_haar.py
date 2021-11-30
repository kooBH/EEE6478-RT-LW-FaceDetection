import cv2

class opencv_haar :
    def __init__(self):
    # Load the cascade
        self.face_cascade = cv2.CascadeClassifier('../pretrained/haarcascade_frontalface_default.xml')

    def detect(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
             cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img
     