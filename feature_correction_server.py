from flask import Flask
import dlib
import numpy as np
import cv2
import pandas as pd
import os
import logging
import requests
from datetime import datetime, time,timezone,timedelta
import csv
from flask import Flask, jsonify, request
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
import shutil
import base64,io
from PIL import Image, ImageOps




app = Flask(__name__)
CORS(app) 

@app.route('/')
def hello_world():  
    current_datetime = datetime.now(timezone.utc)  # Get the current datetime in UTC
    current_time_str = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')  # Format it as a string
    print(current_time_str)
    return 'Hello, World     4 5!'


@app.route('/flaskapi/', methods=['GET'])
def test_api():
    return {"message": "Flask API is Working!"}, 200




@app.route('/sakthi', methods=['GET'])
def sakthi():
    print("Hello, Sakthi!")
    return jsonify({"message": "Hello, Sakthi!"})



@app.route("/traning", methods=["POST"])
def training():  
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Get JSON data from request
    data = request.get_json()
    if not data or "company_id" not in data or "employee" not in data:
        return jsonify({"error": "Missing company_id or employee name"}), 400

    company_id = data["company_id"]
    en_name = data["employee"]
    emp_image = data["image"]
    logging.info(f"Received company_id: {company_id}")
    logging.info(f"Received en_name: {en_name}")
    logging.info(f"Received emp_image: {emp_image}")

    path_images_from_camera = f"data/data_faces_from_camera/{company_id}"

    # Use Dlib's frontal face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
    face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

    # MySQL database configuration
    db_config = {
        'host': "91.108.104.7",
        'user': 'qsis-team',
        'password': "Qsis##db",
        'database': "QuantumFacio"
    }

    def connect_to_database():
        """Connect to the MySQL database."""
        try:
            connection = mysql.connector.connect(**db_config)
            if connection.is_connected():
                logging.info("Connected to MySQL database")
                return connection
        except Error as e:
            logging.error(f"Error connecting to MySQL database: {e}")
            return None

    def create_features_table(connection):
        """Create a table to store facial features if it doesn't exist."""
        try:
            cursor = connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_features (
                    id INT AUTO_INCREMENT PRIMARY KEY, 
                    company_id VARCHAR(100) NOT NULL, 
                    employee_name VARCHAR(255) NOT NULL,
                    
                    created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_date DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    features TEXT NOT NULL,
                    image TEXT NOT NULL,
                    del_i TINYINT DEFAULT 0
                )
            """)
            connection.commit()
            logging.info("Created or verified 'face_features' table")
        except Error as e:
            logging.error(f"Error creating table: {e}")

    def insert_features(connection, employee_name, features, company_id):
        """Insert facial features into the MySQL database as a comma-separated string."""
        try:
            cursor = connection.cursor()
            features_str = ",".join(map(str, features))
            query = "INSERT INTO face_features (employee_name, features, company_id, image) VALUES (%s, %s, %s, %s)"
            res = cursor.execute(query, (employee_name, features_str, company_id, emp_image))
            connection.commit()
            logging.info(f"Inserted features for employee: {employee_name}")
        except Error as e:
            logging.error(f"Error inserting features into database: {e}")
            
    def delete_faces(company_id, en_name):
        """Delete the folder containing face images after processing."""
        try:
            folder_path = f"data/data_faces_from_camera/{company_id}/{en_name}"
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                return jsonify({"message": "Folder deleted successfully"}), 200
            return jsonify({"message": "Folder not found"}), 404
        except Exception as e:
            logging.error(f"Error deleting folder: {e}")
            return jsonify({"message": "Internal Server Error"}), 500

    def preprocess_image(img):
        """Preprocess the image for better face detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Resize if too large (keep aspect ratio)
        max_dim = 1000
        height, width = gray.shape
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            gray = cv2.resize(gray, (int(width * scale), int(height * scale)))
        
        # Convert back to BGR (dlib expects BGR)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def return_128d_features(path_img):
        """Extracts 128D facial features from an image with improved preprocessing."""
        try:
            img_rd = cv2.imread(path_img)
            if img_rd is None:
                logging.error(f"Failed to read image: {path_img}")
                return np.zeros(128, dtype=np.float64)
                
            # Preprocess the image
            img_processed = preprocess_image(img_rd)
            
            # Detect faces with increased upsample for better detection
            faces = detector(img_processed, 1)
            
            if len(faces) == 0:
                # Try again with upsampling
                faces = detector(img_processed, 2)
                
            if len(faces) != 0:
                shape = predictor(img_processed, faces[0])
                face_descriptor = face_reco_model.compute_face_descriptor(img_processed, shape)
                return face_descriptor
                
            logging.warning(f"No face detected in image: {path_img}")
            return np.zeros(128, dtype=np.float64)
            
        except Exception as e:
            logging.error(f"Error processing image {path_img}: {e}")
            return np.zeros(128, dtype=np.float64)

    def return_features_mean_personX(path_face_personX):
        """Computes the average (mean) of 128D face descriptors for a person."""
        features_list_personX = []
        photos_list = os.listdir(path_face_personX)
        
        if not photos_list:
            logging.warning(f"No images found in directory: {path_face_personX}")
            return np.zeros(128, dtype=np.float64)

        for photo in photos_list:
            if not photo.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            features_128d = return_128d_features(os.path.join(path_face_personX, photo))
            if not np.all(features_128d == 0):  # Skip if no face detected
                features_list_personX.append(features_128d)
            else:
                logging.warning(f"Skipping image {photo} - no face detected")

        if features_list_personX:
            return np.array(features_list_personX).mean(axis=0)
        else:
            logging.error(f"No valid faces found in directory: {path_face_personX}")
            return np.zeros(128, dtype=np.float64)

    def get_existing_employees(connection):
        """Reads existing employees from the database to avoid duplicates."""
        existing_names = set()
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT employee_name FROM face_features WHERE company_id = %s", (company_id,))
            for row in cursor.fetchall():
                existing_names.add(row[0])
        except Error as e:
            logging.error(f"Error fetching existing employees: {e}")
        return existing_names

    def main(company_id, en_name):
        """Main function to process face features for a specific employee."""
        connection = connect_to_database()
        if not connection:
            return "Database connection failed"

        create_features_table(connection)
        existing_names = get_existing_employees(connection)

        # Check if the employee already exists in the database
        if en_name in existing_names:
            logging.info(f"Skipping {en_name} (already in database)")
            return f"Employee {en_name} already exists in the database"

        # Check if the employee folder exists
        employee_folder = os.path.join(path_images_from_camera, en_name)
        if not os.path.exists(employee_folder):
            return f"Employee folder not found: {employee_folder}"

        logging.info(f"Processing employee: {en_name}")
        features_mean_personX = return_features_mean_personX(employee_folder)

        # Only insert if we got valid features
        if not np.all(features_mean_personX == 0):
            insert_features(connection, en_name, features_mean_personX, company_id)
            delete_faces(company_id, en_name)
            result = f"Training completed for employee: {en_name}"
        else:
            result = f"Failed to extract valid features for employee: {en_name}"

        if connection.is_connected():
            connection.close()
            logging.info("MySQL connection closed")

        return result

    # Call the main function with the specific employee name
    result = main(company_id, en_name)
    return jsonify({"message": result})

# Load the face database
# DATABASE_PATH = "data/features_all.csv"

# def load_face_database():
#     if os.path.exists(DATABASE_PATH):
#         csv_rd = pd.read_csv(DATABASE_PATH, header=None)
#         names = csv_rd.iloc[:, 0].tolist()
#         features = csv_rd.iloc[:, 1:].values
#         logging.info(f"Loaded {len(names)} faces from database")
#         return names, features
#     else:
#         logging.warning("Face database not found!")
#         return [], []



# Path of cropped faces
path_images_from_camera = "data/data_faces_from_camera/"

# Load DNN model for face detection
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# Load Dlib models
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# MySQL database configuration
db_config = {
        'host': "91.108.104.7",
        'user': 'qsis-team',
        'password': "Qsis##db",
        'database': "QuantumFacio"
}
# Global variables
face_feature_known_list = []  # List to store known face features
face_name_known_list = []  # List to store known face names
face_images_list =[]
frame_skip = 3
frame_cnt = 0

# Connect to the database and load face features
def connect_to_database():
    """Connect to the MySQL database."""
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            logging.info("Connected to MySQL database")
            return connection
    except Error as e:
        logging.error(f"Error connecting to MySQL database: {e}")
        return None

def get_face_database(company_id):
    """Fetch face details from the MySQL database."""
    connection = connect_to_database()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        # Query to fetch all rows from the face_features table
        query = "SELECT employee_name, features, image FROM face_features WHERE company_id = %s"
        cursor.execute(query, (company_id,))  # Pass company_id as a parameter
        
        rows = cursor.fetchall()
        
        for i in rows:
            print("******",i)

        # print("rows",rows)

        if not rows:
            logging.warning("No faces found in the database!")
            return False

        # Clear existing lists
        face_feature_known_list.clear()
        face_name_known_list.clear()

        # Process each row
        for row in rows:
            employee_name = row[0]  # Employee name
            features_str = row[1]  # Features as a comma-separated string
            images=row[2]

            # print("image",employee_name,"-",images)

            # Convert features string to a NumPy array
            features = np.array([float(x) for x in features_str.split(",")])

            # Append to lists
            face_name_known_list.append(employee_name)
            face_feature_known_list.append(features)
            face_images_list.append(images)
        # print("face_name_known_list",face_name_known_list)
        # print("face_feature_known_list",face_feature_known_list)
        # print("face_images_list",face_images_list)

        logging.info(f"Loaded {len(face_name_known_list)} faces from database")
        return True

    except Error as e:
        logging.error(f"Error fetching data from database: {e}")
        return False

    finally:
        if connection.is_connected():
            connection.close()
            logging.info("MySQL connection closed")

# Load face database when the app starts


def return_euclidean_distance(feature_1, feature_2):
    """Calculate the Euclidean distance between two feature vectors."""
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    return np.linalg.norm(feature_1 - feature_2)

def save_attendance_to_api(name_str,company_id, check_out=False):
    """Save attendance to the API."""
    current_datetime = datetime.now(timezone.utc)  # Get the current datetime in UTC
    current_time_str = current_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')  # Format it as a string
    name, employee_id = name_str.split('_')

    if not check_out:
        # For check-in
        payload = {
            "eat_vEmpId": employee_id,
            "eat_vComID": company_id,
            "eat_vName": name,
            "eat_vCheckIn": current_time_str,
        }
        url = "https://quantumfacio.com/api/employee-attendance/checkin"
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logging.info(f"Check-in recorded for {name} at {current_time_str}.")
        else:
            logging.error(f"Failed to record check-in for {name}: {response.text}")
    else:
        # For check-out
        payload = {
            "eat_vCheckOut": current_time_str,
            "eat_vEmpId": employee_id,
        }
        url = "https://quantumfacio.com/api/employee-attendance/checkout"
        response = requests.patch(url, json=payload)
        if response.status_code == 200:
            logging.info(f"Check-out recorded for {name} at {current_time_str}.")
        else:
            logging.error(f"Failed to record check-out for {name}: {response.text}")



def is_check_in_time(check_in):
    """Check if current time is within check-in hours (9:25 AM - 10:00 AM)."""
     # Convert API-provided check-in time (string) to a datetime object
    check_in_time = datetime.strptime(check_in, "%H:%M").time()
    # Define the check-in window (1 hour after check-in)
    check_in_end = (datetime.strptime(check_in, "%H:%M") + timedelta(hours=1)).time()
    # Get the current time
    current_time = datetime.now().time()
    # Check if current time is within the check-in window
    return check_in_time <= current_time <= check_in_end


def is_check_out_time(check_out):
    """Check if current time is after check-out time (6:30 PM)."""
    print(check_out)
    check_out_str = datetime.strptime(check_out, "%H:%M").time()
    print(check_out_str)
    return datetime.now().time() >= check_out_str

@app.route("/recognize", methods=["POST"])
def recognize_faces():
    # check_in = ""
    # check_out = ""


    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Invalid request"}), 400

    face_features = data["features"]
    company_id =data["company_id"]
    
    # print(company_id)
    recognized_names = []

    recognized_images = []
    
    get_face_database(company_id)

    for feature in face_features:   
        distances = [return_euclidean_distance(feature, known_feature) for known_feature in face_feature_known_list]

        # print("face_name_known_list",face_name_known_list)

        # print("distances",distances)
        
        if distances and min(distances) < 0.4:
            recognized_name = face_name_known_list[distances.index(min(distances))]
            recognize_emp_image= face_images_list[distances.index(min(distances))]
            # print("distances.index(min(distances))",distances.index(min(distances)))
            # print("face_images_list  inside of function ", recognize_emp_image)

            # ind = recognized_name
            # print("recognized_name",recognized_name)
            # print("face_name_known_list",face_name_known_list)
            # print("distances",distances)
            # print("distances.index(min(distances))",distances.index(min(distances)))

            recognized_image=face_images_list[distances.index(min(distances))]
            # print("recognized_image", recognized_image)
        else:
            recognized_name = "Unknown"
            recognize_emp_image=""

        recognized_names.append(recognized_name)
        recognized_images.append(recognize_emp_image)
        
       
        # If recognized, save attendance (Check-in during allowed time, otherwise check-out)
        if recognized_name != "Unknown":
            
            url = f"https://quantumfacio.com/api/company/get-company-by-id/{company_id}"
            response = requests.get(url)
            
            print("response",response)

            # Print API response status and data
            print(f"Response Status Code: {response.status_code}")
            try:
                response_data = response.json()  # Convert response to JSON
                company_info = response_data.get("response", {})
                check_in_time = company_info.get("c_vCheckInTime")
                
                check_out_time = company_info.get("c_vCheckOutTime")
                check_in = check_in_time
                check_out = check_out_time

                print("Check-In Time:", check_in_time)
                print("Check-Out Time:", check_out_time)
            except requests.exceptions.JSONDecodeError:
                print("Failed to decode JSON response:", response.text)
            
            if is_check_in_time(check_in): 
                save_attendance_to_api(recognized_name,company_id, check_out=False)
                # print("inside of checkin")
            elif is_check_out_time(check_out):
                save_attendance_to_api(recognized_name,company_id, check_out=True)
                # print("inside of checkOut")

    res=jsonify({"names": recognized_names,"images":recognized_images})


    return res

  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)