import streamlit as st
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from PIL import Image
import numpy as np

# Load the YOLO model
model = YOLO("yolov8l.pt")

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

# Custom CSS for better styling
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f2f6;
        }
        .title {
            color: #4A90E2;
            text-align: center;
            font-size: 3em;
            margin-bottom: 0.5em;
        }
        .subtitle {
            color: #4A90E2;
            text-align: center;
            font-size: 1.5em;
            margin-bottom: 1em;
        }
        .upload-section, .webcam-section {
            background: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            border-radius: 5px;
            font-size: 1em;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Real-Time Object Detection with YOLOv8</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image or use your webcam for real-time object detection.</p>', unsafe_allow_html=True)

# Function to process image
def process_image(image):
    img = np.array(image)
    results = model(img, stream=True)

    people_count = 0
    other_objects_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Display bounding box
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

    return img

# Upload Image
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Processing...")
    processed_image = process_image(image)
    st.image(processed_image, caption='Processed Image.', use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Real-Time Webcam Detection
st.markdown('<div class="webcam-section">', unsafe_allow_html=True)
use_webcam = st.checkbox('Use Webcam')
if use_webcam:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    prev_frame_time = 0

    while True:
        success, img = cap.read()
        if not success:
            st.write("Failed to capture image from webcam.")
            break

        processed_image = process_image(img)
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Convert processed image back to RGB for displaying in Streamlit
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(processed_image_rgb, caption=f'Webcam (FPS: {fps:.2f})', use_column_width=True)

        time.sleep(0.1)

        # This is needed to allow Streamlit to refresh the image display
        st.experimental_rerun()
else:
    st.write("Enable the webcam to start real-time object detection.")
st.markdown('</div>', unsafe_allow_html=True)
