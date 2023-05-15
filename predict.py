# Import Library
from ultralytics import YOLO
import cv2
import time

# Variabel Model
model = YOLO("best.pt")

# Open Kamera 
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise("No Camera")

while True :
    ret, image = cam.read()
    if not ret:
        break
    _time_mulai = time.time()
    result = model.predict(image, show=True)

    print("Waktu", time.time()-_time_mulai)
    # Cv2.imshow("image", image)
    _key = cv2.waitKey(1)
    if _key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()