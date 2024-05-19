import os
from flask import Flask, render_template, Response, send_from_directory
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

app = Flask(__name__)

# Use a smaller YOLO model for faster inference
model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "auto rickshaw"
              ]

def gen():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    process_every_n_frames = 2  # Process every 2nd frame

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % process_every_n_frames == 0:
            results = model(img, stream=True)

            people_count = 0
            other_objects_count = 0

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    cvzone.cornerRect(img, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    object_name = classNames[cls]

                    if object_name == "person":
                        people_count += 1
                    else:
                        other_objects_count += 1

                    cv2.putText(img, f'{object_name} {conf}', (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(img, f'People: {people_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2, cv2.LINE_AA)
            cv2.putText(img, f'Other Objects: {other_objects_count}', (img.shape[1] - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (100, 100, 100), 2, cv2.LINE_AA)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(f"FPS: {fps}")

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
