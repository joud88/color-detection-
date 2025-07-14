
# 🎨 Color Detection

Color Detection is a Python tool that uses OpenCV to identify and highlight red, green, blue, yellow, white, and black areas in images using HSV color filtering. The program identifies color regions and draws bounding boxes in the matching color for clear visualization. 

## 🧾 Features

- Detects the following colors using HSV filtering:
  - Red
  - Green
  - Blue
  - Yellow
  - White
  - Black
- Draws labeled bounding boxes in the detected color
- Outputs a result image (`output_cd.png`) for visualization

## 📂 Files

| File             | Description                                  |
|------------------|----------------------------------------------|
| `color_detection.py` | Main script for color detection           |
| `image1.jpg`         | Input image used for detection            |
| `output_cd.png`      | Output image with detected color boxes    |


## ▶️ How to Run

| Step | Instruction                                                                 |
|------|------------------------------------------------------------------------------|
| 1️⃣   | Make sure you have Python installed                                         |
| 2️⃣   | Install OpenCV with `pip install opencv-python`                            |
| 3️⃣   | Place your input image as `image1.jpg` in the same folder                  |
| 4️⃣   | Run the script using `python color_detection.py`                           |
| 5️⃣   | The result will be shown in a window and saved as `output_cd.png`          |


