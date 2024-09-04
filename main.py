import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

# Read image
image_path = "/content/number plate.webp"
img = cv2.imread(image_path)

# Check if the image was loaded correctly
if img is None:
    print("Error: Could not load image. Please check the file path.")
else:
    # Convert image from BGR (OpenCV default) to RGB (EasyOCR expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Instance text detector
    reader = easyocr.Reader(['en'], gpu=False)

    # Detect text on image
    text_ = reader.readtext(img_rgb)

    threshold = 0.25
    # Draw bounding box and text
    for t_, t in enumerate(text_):
        print(t)

        bbox, text, score = t

        if score > threshold:
        
            cv2.rectangle(img, tuple(bbox[0]), tuple(bbox[2]), (255, 0, 0), 5)
            cv2.putText(img, text, tuple(bbox[0]), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 4)

    # Display the image with the detected text
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


   
