import cv2
import numpy as np

# Load the image
image = cv2.imread('image1.jpg')

# Check if image loaded
if image is None:
    print("❌ Couldn't find 'image1.jpg'. Make sure it's in the same folder.")
    exit()

# Convert image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define HSV color ranges
color_ranges = {
    "Red":    [(0, 100, 100), (10, 255, 255)],
    "Green":  [(40, 50, 50), (80, 255, 255)],
    "Blue":   [(100, 150, 0), (140, 255, 255)],
    "Yellow": [(20, 100, 100), (30, 255, 255)],
    "White":  [(0, 0, 200), (180, 30, 255)],
    "Black":  [(0, 0, 0), (180, 255, 30)],
}

# Matching BGR colors for display (for box/text)
display_colors = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
}

# Detect and draw
for color_name, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            color = display_colors[color_name]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

# Show the result
cv2.imshow('Detected Colors', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
