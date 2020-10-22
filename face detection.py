import cv2
import numpy as np


face_classifier = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:

	ret, frame = cap.read()

	if ret:
		frame = cv2.flip(frame, 1)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_classifier.detectMultiScale(gray)
		for fx, fy, fw, fh in faces:
			roi_gray = gray[fy : fy+fh, fx : fx+fw]
			roi_color = frame[fy : fy+fh, fx : fx+fw]
			cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 255), 3)
			
			eyes = eye_classifier.detectMultiScale(roi_gray)
			for ex, ey, ew, eh in eyes:
				roi_eyes = roi_gray[ey : ey+eh, ex : ex+ew]
				cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 4)
		cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)	# creating a resizable window
		cv2.imshow("Capture", frame)

	key = cv2.waitKey(1)

	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
