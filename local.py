import dlib
import numpy as np
import cv2
import requests
from datetime import datetime
import spidev
import RPi.GPIO as GPIO
from PIL import Image, ImageDraw, ImageFont
import time as tt
from io import BytesIO
import threading
import logging

# GPIO setup for SPI display
DC = 24  # Data/Command pin
RST = 25  # Reset pin
CS = 8  # Chip Select pin (for display)

GPIO.setmode(GPIO.BCM)
GPIO.setup(DC, GPIO.OUT)
GPIO.setup(RST, GPIO.OUT)
GPIO.setup(CS, GPIO.OUT)

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 20000000  # 20 MHz

# ILI9486 commands
ILI9486_SWRESET = 0x01
ILI9486_SLPOUT = 0x11
ILI9486_DISPON = 0x29
ILI9486_CASET = 0x2A
ILI9486_PASET = 0x2B
ILI9486_RAMWR = 0x2C

# Display dimensions
LCD_WIDTH = 320  # Display width
LCD_HEIGHT = 480  # Display height
IMAGE_HEIGHT = 240  # Height allocated for the image
TEXT_HEIGHT = 240  # Height allocated for the text

# Global flag to control video playback
video_running = False
video_thread = None

def send_command(cmd):
    GPIO.output(DC, GPIO.LOW)
    GPIO.output(CS, GPIO.LOW)
    spi.xfer([cmd])
    GPIO.output(CS, GPIO.HIGH)

def send_data(data):
    GPIO.output(DC, GPIO.HIGH)
    GPIO.output(CS, GPIO.LOW)
    spi.xfer([data])
    GPIO.output(CS, GPIO.HIGH)

def set_window(x0, y0, x1, y1):
    send_command(ILI9486_CASET)
    send_data(x0 >> 8)
    send_data(x0 & 0xFF)
    send_data(x1 >> 8)
    send_data(x1 & 0xFF)
    send_command(ILI9486_PASET)
    send_data(y0 >> 8)
    send_data(y0 & 0xFF)
    send_data(y1 >> 8)
    send_data(y1 & 0xFF)

def init_display():
    GPIO.output(RST, GPIO.LOW)
    tt.sleep(1)
    GPIO.output(RST, GPIO.HIGH)
    tt.sleep(1)
    send_command(ILI9486_SWRESET)
    tt.sleep(1)
    send_command(ILI9486_SLPOUT)
    tt.sleep(1)
    send_command(ILI9486_DISPON)
    tt.sleep(1)
    send_command(0x36)
    send_data(0x48)
    send_command(0x3A)
    send_data(0x55)

def rgb_to_rgb565(r, g, b):
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)

def display_image(image_input):
    # Resize the image to fit the display (320x480)
    image = image_input.resize((LCD_WIDTH, LCD_HEIGHT)).convert("RGB")
    image_rgb565 = []
    
    for y in range(LCD_HEIGHT):
        for x in range(LCD_WIDTH):
            r, g, b = image.getpixel((x, y))
            rgb565 = rgb_to_rgb565(r, g, b)
            image_rgb565.append(rgb565 >> 8)
            image_rgb565.append(rgb565 & 0xFF)
    
    # Set the window for the entire display
    set_window(0, 0, LCD_WIDTH - 1, LCD_HEIGHT - 1)
    send_command(ILI9486_RAMWR)
    
    GPIO.output(DC, GPIO.HIGH)
    GPIO.output(CS, GPIO.LOW)
    
    CHUNK_SIZE = 4096
    for i in range(0, len(image_rgb565), CHUNK_SIZE):
        spi.writebytes(image_rgb565[i:i + CHUNK_SIZE])
    
    GPIO.output(CS, GPIO.HIGH)

def display_photo_and_info(image, name, id_value, time, date):
    """Display photo in top half and info in bottom half of the screen"""
    # Resize image to fit the top half
    image_resized = image.resize((LCD_WIDTH, IMAGE_HEIGHT)).convert("RGB")
    
    # Create the full display image (combining photo and info)
    full_image = Image.new("RGB", (LCD_WIDTH, LCD_HEIGHT), (0, 0, 0))
    full_image.paste(image_resized, (0, 0))
    
    # Add text info to the bottom half
    draw = ImageDraw.Draw(full_image)
    
    # Load a bold font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font
    
    # Draw the name, ID, time, and date
    text_y = IMAGE_HEIGHT + 10
    draw.text((10, text_y), f"Name: {name}", font=font, fill=(255, 255, 255))
    text_y += 30
    draw.text((10, text_y), f"ID: {id_value}", font=font, fill=(255, 255, 255))
    text_y += 30
    draw.text((10, text_y), f"Time: {time}", font=font, fill=(255, 255, 255))
    text_y += 30
    draw.text((10, text_y), f"Date: {date}", font=font, fill=(255, 255, 255))
    
    # Convert the image to RGB565 format and display
    image_rgb565 = []
    for y in range(LCD_HEIGHT):
        for x in range(LCD_WIDTH):
            r, g, b = full_image.getpixel((x, y))
            rgb565 = rgb_to_rgb565(r, g, b)
            image_rgb565.append(rgb565 >> 8)
            image_rgb565.append(rgb565 & 0xFF)
    
    # Set the window for the entire display
    set_window(0, 0, LCD_WIDTH - 1, LCD_HEIGHT - 1)
    send_command(ILI9486_RAMWR)
    
    GPIO.output(DC, GPIO.HIGH)
    GPIO.output(CS, GPIO.LOW)
    
    CHUNK_SIZE = 4096
    for i in range(0, len(image_rgb565), CHUNK_SIZE):
        spi.writebytes(image_rgb565[i:i + CHUNK_SIZE])
    
    GPIO.output(CS, GPIO.HIGH)

def clear_display():
    """Clear the entire LCD display by filling it with white color."""
    # Create a blank white image for the entire display (320x480)
    white_image = Image.new("RGB", (LCD_WIDTH, LCD_HEIGHT), (255, 255, 255))
    
    # Convert the image to RGB565 format
    image_rgb565 = []
    for y in range(LCD_HEIGHT):
        for x in range(LCD_WIDTH):
            r, g, b = white_image.getpixel((x, y))
            rgb565 = rgb_to_rgb565(r, g, b)
            image_rgb565.append(rgb565 >> 8)
            image_rgb565.append(rgb565 & 0xFF)
    
    # Set the window for the entire display
    set_window(0, 0, LCD_WIDTH - 1, LCD_HEIGHT - 1)
    send_command(ILI9486_RAMWR)
    
    GPIO.output(DC, GPIO.HIGH)
    GPIO.output(CS, GPIO.LOW)
    
    CHUNK_SIZE = 4096
    for i in range(0, len(image_rgb565), CHUNK_SIZE):
        spi.writebytes(image_rgb565[i:i + CHUNK_SIZE])
    
    GPIO.output(CS, GPIO.HIGH)

def stop_video():
    """Function to stop the currently playing video"""
    global video_running
    video_running = False
    # Wait for video thread to finish if it exists
    if video_thread is not None and video_thread.is_alive():
        video_thread.join(1)  # Wait up to 1 second for the thread to finish

def play_video_once(video_path):
    """Play a video just once, blocking until it completes"""
    global video_running, video_thread
    
    # Stop any currently running video first
    stop_video()
    
    # Set the flag to True for the new video
    video_running = True
    
    # Create a completion event
    video_completed = threading.Event()
    
    def _play_video():
        global video_running
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            video_completed.set()
            return
            
        while cap.isOpened() and video_running:
            ret, frame = cap.read()
            if not ret:
                # Video ended
                break
                    
            # Resize the frame to fit the full display (320x480)
            frame = cv2.resize(frame, (LCD_WIDTH, LCD_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            display_image(image)
        
        cap.release()
        video_completed.set()
    
    # Start the video playback in a new thread
    video_thread = threading.Thread(target=_play_video)
    video_thread.daemon = True  # Daemonize thread to exit when the main program exits
    video_thread.start()
    
    # Wait for video to complete
    video_completed.wait()

def display_video(video_path, loop=True):
    """Play a video in a separate thread."""
    global video_running, video_thread
    
    # Stop any currently running video first
    stop_video()
    
    # Set the flag to True for the new video
    video_running = True
    
    def _play_video():
        global video_running
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return
            
        while cap.isOpened() and video_running:
            ret, frame = cap.read()
            if not ret:
                if loop and video_running:
                    # Rewind the video if looping is enabled
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
                    
            # Resize the frame to fit the full display (320x480)
            frame = cv2.resize(frame, (LCD_WIDTH, LCD_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            display_image(image)
        
        cap.release()
    
    # Start the video playback in a new thread
    video_thread = threading.Thread(target=_play_video)
    video_thread.daemon = True  # Daemonize thread to exit when the main program exits
    video_thread.start()
    
    return video_thread

# Load DNN model for face detection
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Load Dlib model for feature extraction
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

SERVER_URL = " http://127.0.0.1:5000/recognize"  # Replace with your Flask server IP

class FaceRecognizerClient:
    def __init__(self):
        self.frame_skip = 3 # Process every 5th frame
        self.frame_cnt = 0
        self.recognition_active = True

    def detect_faces(self, img):
        blob = cv2.dnn.blobFromImage(img, 1.0, (150, 150), (104.0, 177.0, 123.0))  # Smaller input size
        face_net.setInput(blob)
        detections = face_net.forward()
        h, w = img.shape[:2]
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
        return faces

    def extract_face_features(self, img, faces):
        face_descriptors = []
        for (startX, startY, endX, endY) in faces:
            shape = predictor(img, dlib.rectangle(startX, startY, endX, endY))
            face_descriptor = np.array(face_reco_model.compute_face_descriptor(img, shape))
            face_descriptors.append(face_descriptor.tolist())
        return face_descriptors

    def get_image_from_url(self, url):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Ensure the request was successful
            image = Image.open(BytesIO(response.content))
            return image
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading image: {e}")
            return None

    def send_features_to_server(self, features):
        if not features:
            # Only display "Show Your Face" if we're actively scanning
            if self.recognition_active:
                display_video("face.mp4", loop=True)
            return []
        
        try:
            com = 'QF20250320-1'
            response = requests.post(SERVER_URL, json={"features": features, "company_id": com})
            response_data = response.json()
            
            names = response_data.get("names", [])
            img_urls = response_data.get("images", [])

            if names and self.recognition_active:
                str_name = names[0]
                
                if "_" in str_name:
                    # Split the name to get individual components
                    name, id_value = str_name.split("_", 1)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    current_date = datetime.now().strftime("%Y-%m-%d")

                    if img_urls:
                        # Temporarily pause face recognition process
                        self.recognition_active = False
                        
                        # 1. First, stop the currently playing face.mp4 video
                        stop_video()
                        
                        # 2. Download and display the user's image with details
                        user_image = self.get_image_from_url(img_urls[0])
                        if user_image:
                            display_photo_and_info(user_image, name, id_value, current_time, current_date)
                            tt.sleep(5)  # Display for 3 seconds
                        
                        # # 3. Play the tic.mp4 video once
                        # play_video_once("tic.mp4")
                        
                        # 4. Restart the face recognition video
                        display_video("face.mp4", loop=True)
                        
                        # 5. Resume face recognition
                        self.recognition_active = True
                    
            return names
        except requests.exceptions.RequestException as e:
            logging.error(f"Error connecting to server: {e}")
            return []

    def process_video(self):
        # Start with Face_scan video first
        display_video("face.mp4", loop=True)
        
        # Allow time for the video to start
        tt.sleep(1)

        cap = cv2.VideoCapture(0)

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        # Try setting width and height
        width, height = 480, 360  # Smaller resolution for faster processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Verify if resolution was set successfully
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width != width or actual_height != height:
            print(f"Warning: Camera resolution not set correctly, using available resolution: {actual_width}x{actual_height}")

        try:
            while cap.isOpened():
                self.frame_cnt += 1
                ret, frame = cap.read()
                if not ret:
                    break

                # Only process every nth frame based on frame_skip
                if self.frame_cnt % self.frame_skip == 0 and self.recognition_active:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    faces = self.detect_faces(small_frame)

                    if faces:
                        features = self.extract_face_features(small_frame, faces)
                        names = self.send_features_to_server(features)

                        for (box, name) in zip(faces, names if names else ["Unknown"] * len(faces)):
                            (startX, startY, endX, endY) = box
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame, (startX * 2, startY * 2), (endX * 2, endY * 2), color, 2)
                            cv2.putText(frame, name, (startX * 2, startY * 2 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                cv2.imshow("Face Recognition", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            # Cleanup before exiting
            stop_video()  # Stop any running video
            cap.release()
            cv2.destroyAllWindows()
            clear_display()  # Clear the entire LCD to a white screen
            GPIO.cleanup()  # Clean up GPIO resources

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_display()
    client = FaceRecognizerClient()
    client.process_video()