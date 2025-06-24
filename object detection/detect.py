from ultralytics import YOLO
import cv2


model = YOLO("yolov5su.pt")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("✅ Webcam opened successfully. Starting detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break


    results = model.predict(source=[frame], stream=True, conf=0.4)


    for r in results:
        annotated_frame = r.plot()


    cv2.imshow("YOLOv5 Real-Time Detection", annotated_frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
