YOLOv8 Object Detection
This guide explains how to use a computer program to find objects in a picture using a tool called YOLOv8.
Step 1: Install Python
Python is a programming language we use to write our instructions (code). To get it:
1. Open your web browser and go to https://www.python.org/downloads/
2. Click on the big yellow button to download Python (we used version 3.13.3).
3. Open the downloaded file and follow the steps to install it.
4. Make sure you check the box that says 'Add Python to PATH' before clicking install.
Step 2: Set Up Your Work Environment
A virtual environment is like a clean workspace for your project. It keeps things organized and avoids problems.
1. Open a code editor like Visual Studio Code.
2. Open a terminal or command prompt window in your project folder.
3. Type the following command and press Enter:
python -m venv venv
This creates a new folder called 'venv'. It contains your workspace setup.
Now activate it by typing:
venv\Scripts\activate
If done correctly, you'll see (venv) appear in your terminal.
Step 3: Install the YOLO Tool and Others
YOLO is the tool that finds objects in pictures. To install it:
1. Make sure your virtual environment is activated.
2. Type this command and press Enter:
pip install ultralytics
This will install YOLO and other tools it needs to work. It may take a few moments.
Step 4: Write the Code to Find Objects in an Image
Now you'll write a short program to load an image and let YOLO find things like people or cars in it.
Open a new file in your code editor and name it detect.py.
Copy and paste the following code into it:

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont

model = YOLO("yolov8n.pt")
results = model("bus.jpg")
num_predictions = len(results[0].boxes)
img_pil = Image.open("bus.jpg")
draw = ImageDraw.Draw(img_pil)
for i, box in enumerate(results[0].boxes.xyxy):
    conf = results[0].boxes.conf[i] * 100
    x1, y1, x2, y2 = box.tolist()
    text = f"{conf:.2f}%"
    font = ImageFont.load_default()
    text_width, text_height = font.getbbox(text)[2:]
    bg_x1, bg_y1, bg_x2, bg_y2 = x1, y1 - text_height - 5, x1 + text_width + 5, y1
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill="black")
    draw.text((x1 + 2, y1 - text_height - 3), text, fill="white", font=font)
    draw.rectangle(box.tolist(), outline="yellow", width=3)
img_pil.save("Final.jpg")

Here’s what this code does:
- It loads the YOLO model (the brain that finds objects).
- It opens an image file called 'bus.jpg'.
- It checks how many objects YOLO found in the image.
- For each object, it draws a yellow box around it.
- It also shows how confident YOLO is about what it found (as a percentage).
- Finally, it saves the new image as 'Final.jpg' with the boxes and confidence shown.
Step 5: Run Your Program
Now that everything is ready:
1. Make sure your files are in the same folder: detect.py, yolov8n.pt, and bus.jpg.
2. Open your terminal, and run the program by typing:
python detect.py
3. After a few seconds, a new image named 'Final.jpg' will appear.
Open it and you’ll see boxes around the objects YOLO found.
