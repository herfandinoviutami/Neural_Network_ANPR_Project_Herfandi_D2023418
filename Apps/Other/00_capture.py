import cv2 as cv
from ultralytics import YOLO
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import matplotlib.image as mpimg

model = YOLO(r"models/plat_model.pt")
detect_chars = tf.keras.models.load_model("models/char_model.h5")
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
image = cv.imread(r"images\EA4529AM.jpg")
image = cv.resize(image, (int(image.shape[1] * 0.4), int(image.shape[0] * 0.4)))
classNames = ["license-plate"]
results = model(image)[0]
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    rects = cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv.imwrite("process/black/01_rects.jpg", rects)

    croppedimg = image[y1:y2, x1:x2]
    cv.imwrite("process/black/02_crop_rects.jpg", croppedimg)

    gray = cv.cvtColor(croppedimg, cv.COLOR_BGR2GRAY)
    if np.mean(gray) > 128:
        # white plate
        gray = 255 - gray
    cv.imwrite("process/black/03_gray.jpg", gray)

    _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite("process/black/04_binary.jpg", binary)

    erode = cv.erode(binary, (3, 3))
    cv.imwrite("process/black/05_erode.jpg", erode)

    dilate = cv.erode(erode, (3, 3))
    cv.imwrite("process/black/06_dilate.jpg", dilate)

    contours, _ = cv.findContours(erode, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contour: cv.boundingRect(contour)[0])
    image_copy = croppedimg.copy()

    platno = []
    num_plate = []
    index = 0
    digit_w, digit_h = 40, 40
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        ratio = h / w
        if h / croppedimg.shape[0] >= 0.3 and 1 <= ratio <= 3.5:
            platno.append(index)
            cv.drawContours(image_copy, contours, index, (0, 255, 0), 3)

            cv.imwrite("process/black/07_contour.jpg", image_copy)
        #           print(f'x: {x}, y: {y}, w: {w}, h: {h}')
        index += 1

    for index in platno:
        x, y, w, h = cv.boundingRect(contours[index])
        cv.rectangle(croppedimg, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # potong citra karakter
        char_crop = cv.cvtColor(dilate[y : y + h, x : x + w], cv.COLOR_GRAY2BGR)
        # resize citra karakternya
        char_crop = cv.resize(char_crop, (digit_w, digit_h))
        # cv.imwrite('index.jpg', char_crop[0])
        # preprocessing citra ke numpy array
        img_array = tf.keras.preprocessing.image.img_to_array(char_crop)

        # agar shape menjadi [1, h, w, channels]
        img_array = tf.expand_dims(img_array, 0)

        # buat prediksi
        predictions = detect_chars.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        num_plate.append(class_names[np.argmax(score)])
    # print(class_names[np.argmax(score)], end='')

    # Gabungkan string pada list
    plate_number = ""
    for a in num_plate:
        plate_number += a

    # # Hasil deteksi dan pembacaan
    # print(plate_number)
    confidence = math.ceil((box.conf[0] * 100)) / 100
    print("Confidence --->", confidence)

    # Get class name
    cls = int(box.cls[0])
    print("Class name -->", classNames[cls])

    # Prepare text for display
    text = f"{classNames[cls]} {confidence}% | {plate_number}"

    # Object details
    org = [x1, y1]
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1

    # Add background for text
    (text_width, text_height) = cv.getTextSize(
        text, font, fontScale=fontScale, thickness=thickness
    )[0]
    box_coords = ((org[0], org[1]), (org[0] + text_width + 2, org[1] - text_height - 2))
    cv.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv.FILLED)

    # Put text on image
    cv.putText(image, text, org, font, fontScale, color, thickness)
    print("Number plate detected -->", plate_number)
# Save the image with text and bounding box
cv.imwrite("process/black/08_final_result.jpg", image)

img = mpimg.imread("process/black/08_final_result.jpg")

# Display the image
plt.imshow(img)
plt.show()
