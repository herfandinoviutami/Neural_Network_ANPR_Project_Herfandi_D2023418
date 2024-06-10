from ultralytics import YOLO
import cv2 as cv
import math
import numpy as np
import tensorflow as tf

# import time
import csv

# Create a CSV file and write the header
with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Text", "Confidence"])


# start webcam
cap = cv.VideoCapture(r"test\WhatsApp Video 2024-05-26 at 18.29.45.mp4")
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# Membuat objek VideoWriter
fourcc = cv.VideoWriter_fourcc(*"mp4v")  # atau gunakan 'XVID'
out = cv.VideoWriter("videos/resultsss.mp4", fourcc, fps, (width, height))
# model
model = YOLO(r"models\plat_model.pt")
char_detect = tf.keras.models.load_model(r"models\char_model.h5")
# object classes
classNames = ["License-plate"]
class_names = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

frame_count = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    frame_count += 1
    # num_plate = []
    # num_plate.clear
    # PAUSE 2 SECONDS BEFORE CONTINUING TO THE NEXT LOOP
    # time.sleep(2)

    # coordinates
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
            # processing
            plate_img = img[y1:y2, x1:x2]
            rects = cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # if frame_count % 60 == 0:
            #     cv.imwrite(f'frame_{frame_count}.jpg', rects)
            # cv.imwrite('process/plate/plate.jpg', rects)
            gray = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
            if np.mean(gray) > 128:
                # white plate
                gray = 255 - gray
            _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            erode = cv.erode(binary, (4, 4))
            dilate = cv.dilate(erode, (3, 3))
            # if frame_count % 10 == 0:
            #     cv.imwrite(f'process/plate/plate_{frame_count}.jpg', dilate)
            contours, _ = cv.findContours(
                erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(contours, key=lambda contour: cv.boundingRect(contour)[0])
            image_copy = plate_img.copy()

            # find coutour

            digit_w, digit_h = 40, 40
            num_plate = []  # Clear the list for each new plat
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                ratio = h / w

                if 1 <= ratio <= 3.5:
                    if h / plate_img.shape[0] >= 0.3:
                        cv.rectangle(
                            plate_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3
                        )
                        if frame_count % 10 == 0:
                            cv.imwrite(
                                f"process/segmentasi/frame_segmentasi_{frame_count}.jpg",
                                plate_img,
                            )
                        # #potong citra karakter
                        char_crop = cv.cvtColor(
                            dilate[y : y + h, x : x + w], cv.COLOR_GRAY2BGR
                        )
                        # resize citra karakternya
                        char_crop = cv.resize(char_crop, (digit_w, digit_h))
                        # preprocessing citra ke numpy array
                        img_array = tf.keras.preprocessing.image.img_to_array(char_crop)

                        # agar shape menjadi [1, h, w, channels]
                        if frame_count % 10 == 0:
                            img_array = tf.expand_dims(img_array, 0)
                            predictions = char_detect.predict(img_array)
                            score = tf.nn.softmax(predictions[0])
                            num_plate.append(class_names[np.argmax(score)])
                            plate_number = ""
                            for a in num_plate:
                                plate_number += a
            # put box in cam
            cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            if frame_count % 10 == 0:
                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                text = f"{classNames[cls]} {confidence}%"
                # ...
                # Write to CSV
                with open("output.csv", "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_count, "".join(num_plate), confidence])
                # object details
                org = [x1, y1]
                font = cv.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (255, 0, 0)
                thickness = 1

                # add background for text
                (text_width, text_height) = cv.getTextSize(
                    text, font, fontScale=fontScale, thickness=thickness
                )[0]
                box_coords = (
                    (org[0], org[1]),
                    (org[0] + text_width + 2, org[1] - text_height - 2),
                )
                cv.rectangle(
                    img, box_coords[0], box_coords[1], (255, 255, 255), cv.FILLED
                )

                # put text on top of the background
                cv.putText(img, text, org, font, fontScale, color, thickness)
                cv.imshow("License Plate", plate_img)
                if frame_count % 10 == 0:
                    cv.imwrite(f"process/plate/frame_YOLO_{frame_count}.jpg", rects)
                print("Number plate detected --> ", "".join(num_plate))

    # # Hasil deteksi dan pembacaanprint(plate_number)

    cv.imshow("Webcam", img)
    out.write(img)

    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
