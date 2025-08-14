from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Predict on an image
results = model("bus.jpg")

# Get the number of detected objects
num_predictions = len(results[0].boxes)

# Option 2: Display Bounding Boxes using PIL with Background for Text
img_pil = Image.open("bus.jpg")
draw = ImageDraw.Draw(img_pil)

for i, box in enumerate(results[0].boxes.xyxy):
    conf = results[0].boxes.conf[i] * 100  # Convert confidence to percentage
    x1, y1, x2, y2 = box.tolist()
    text = f"{conf}"
    # Define text box background
    text = f"{conf:.2f}%"
    font = ImageFont.load_default()
    text_width, text_height = font.getbbox(text)[2:]  # Extract width and height
    
    # Draw rectangle for background
    bg_x1, bg_y1, bg_x2, bg_y2 = x1, y1 - text_height - 5, x1 + text_width + 5, y1
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill="black")  # Background color

    # Draw text on top of background
    draw.text((x1 + 2, y1 - text_height - 3), text, fill="white", font=font)

    # Draw bounding box
    draw.rectangle(box.tolist(), outline="yellow", width=3)

# Show the final image
img_pil.save("Final.jpg")

