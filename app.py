from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import joblib
from dotenv import load_dotenv
import os
import logging
import json
import numpy as np
import pytz

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG to get more detailed output

# Load category list from JSON file
CATEGORY_LIST_FILE = 'categoryList.json'

def load_category_list():
    try:
        if os.path.exists(CATEGORY_LIST_FILE):
            with open(CATEGORY_LIST_FILE, 'r') as file:
                return json.load(file)
    except Exception as e:
        logging.error(f"Error loading category list: {e}")
    return {
        "Core Productive": [],  # Added Core Productive
        "Productive": [],
        "Unproductive": [],
        "Idle": [],
        "Others": []
    }

category_list = load_category_list()

def save_category_list():
    try:
        with open(CATEGORY_LIST_FILE, 'w') as file:
            json.dump(category_list, file, indent=4)
        logging.info("Category list updated and saved.")
    except Exception as e:
        logging.error(f"Error saving category list: {e}")


# Load machine learning models
try:
    logging.info("Loading machine learning models...")
    label_encoder = joblib.load('label_encoder.joblib')
    vectorizer = joblib.load('tfif_vectorizer.joblib')
    model = joblib.load('website_classifier_model.joblib')
    logging.info("Machine learning models loaded.")
except Exception as e:
    logging.error(f"Error loading machine learning models: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Database configuration
try:
    logging.info("Setting up database connection...")
    engine = create_engine(f'postgresql+psycopg2://{os.getenv("DB_USERNAME")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}/{os.getenv("DB_NAME")}')
    Session = sessionmaker(bind=engine)
    logging.info("Database connection established.")
except Exception as e:
    logging.error(f"Error setting up database connection: {e}")

def get_all_device_ids(organization_uid):
    logging.info("Fetching all device IDs and user names for organization UID: {organization_uid}...")
    session = Session()
    try:
        result = session.execute(text("SELECT device_uid, user_name FROM devices WHERE organization_uid = :organization_uid"), {"organization_uid": organization_uid}).fetchall()
        devices = [{'device_uid': row[0], 'user_name': row[1]} for row in result]
        logging.info(f"Fetched {len(devices)} devices for organization UID: {organization_uid}.")
        return devices
    except Exception as e:
        logging.error(f"Error fetching device IDs: {e}")
        return []
    finally:
        session.close()

def convert_to_z_format(db_timestamp):
    # Parse the database timestamp
    dt = datetime.strptime(db_timestamp, '%Y-%m-%d %H:%M:%S')
    
    # Convert to UTC time
    utc_time = pytz.utc.localize(dt)
    
    # Convert to 'Z' format
    z_format_time = utc_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    return z_format_time

def convert_to_local_time(z_format_time, timezone_str='Asia/Kolkata'):
    # Parse the 'Z' format time
    utc_time = datetime.strptime(z_format_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    utc_time = pytz.utc.localize(utc_time)
    
    # Convert to the local time zone
    local_tz = pytz.timezone(timezone_str)
    local_time = utc_time.astimezone(local_tz)
    
    # Format the local time
    local_time_str = local_time.strftime('%Y-%m-%d %H:%M:%S')
    
    return local_time_str

def get_user_activities(device_uid, date):
    logging.info(f"Fetching activities for device UID: {device_uid} on date: {date}...")
    session = Session()
    try:
        result = session.execute(text(
            "SELECT page_title, timestamp FROM user_activity WHERE DATE(timestamp) = :date AND user_uid = :device_uid ORDER BY timestamp"
        ), {"date": date, "device_uid": str(device_uid)}).fetchall()
        
        activities = []
        for row in result:
            page_title = row[0]
            db_timestamp = row[1].strftime('%Y-%m-%d %H:%M:%S')  # Assuming timestamp is datetime object
            
            # Convert database timestamp to 'Z' format
            z_format_time = convert_to_z_format(db_timestamp)
            
            # Convert 'Z' format timestamp to local time format
            local_time = convert_to_local_time(z_format_time)
            
            # Parse local_time back to datetime object for further processing
            local_time_dt = datetime.strptime(local_time, '%Y-%m-%d %H:%M:%S')
            
            activities.append((page_title, local_time_dt))
        
        return activities
    except Exception as e:
        logging.error(f"Error fetching user activities: {e}")
        return []
    finally:
        session.close()

def predict_category(page_title):
    try:
        logging.info(f"Predicting category for page title: {page_title}...")

        # Match against the category list
        for category, keywords in category_list.items():
            if category == "Others":
                continue
            for keyword in keywords:
                if keyword.lower() in page_title.lower():
                    logging.info(f"Matched keyword '{keyword}' in category '{category}'")
                    return category, 1.0, False  # High confidence for manual matches

        # Use ML model if no match is found
        transformed = vectorizer.transform([page_title])
        prediction = model.predict(transformed)
        confidence = np.max(model.predict_proba(transformed))  # Get the highest confidence score
        category = label_encoder.inverse_transform(prediction)[0]
        logging.info(f"Predicted category: {category} with confidence: {confidence}")

        # Handle low confidence or missing category
        if confidence < 0.4 or category not in category_list:
            if page_title not in category_list["Others"]:
                category_list["Others"].append(page_title)
                save_category_list()
            return "Others", 1.0, True  # Assign to 'Others' for ambiguous titles

        return category, confidence, False
    except Exception as e:
        logging.error(f"Error predicting category: {e}")
        return "Neutral", 0.0, False

def map_category_to_productivity(category):
    logging.info(f"Category Maping: {category}")
    try:
        if category in category_list.get("Core Productive", []):
            return "core productive"
        elif category in category_list.get("Productive", []):
            return "productive"
        elif category in category_list.get("Unproductive", []):
            return "unproductive"
        elif category in category_list.get("Idle", []):
            return "idle"
        else:
            logging.warning(f"Category '{category}' not found in category list. Defaulting to 'idle'.")
            return "idle"  # Default to idle for unknown categories
    except Exception as e:
        logging.error(f"Error mapping category to productivity: {e}")
        return "idle"

def calculate_productivity_internal(activities):
    try:
        logging.info("Calculating productivity...")
        total_time = timedelta(hours=1)
        productivity_data = []

        hourly_activities = {}
        for page_title, timestamp in activities:
            logging.info(f"Processing activity - Title: {page_title}, Timestamp: {timestamp}")
            hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_activities:
                hourly_activities[hour_key] = []
                print(f"hour_key: %s" % hour_key)
            hourly_activities[hour_key].append((page_title, timestamp))

        logging.info(f"Hourly activities keys: {list(hourly_activities.keys())}")

        for hour in range(9, 19):  # From 9:00 AM to 7:00 PM
            hour_key = datetime.strptime(f"{hour}:00", "%H:%M").replace(year=timestamp.year, month=timestamp.month, day=timestamp.day)
            if hour_key not in hourly_activities:
                logging.info(f"No activities found for hour: {hour_key}")
                productivity_data.append([
                    {"productivity": "away", "percent": "100%"},
                    {"productivity": "", "percent": ""},
                    {"productivity": "", "percent": ""},
                    {"productivity": "", "percent": ""},
                    {"productivity": "", "percent": ""},
                    {"productivity": "", "percent": ""}
                ])
            else:
                activities = hourly_activities[hour_key]
                activities.sort(key=lambda x: x[1])
                first_activity = activities[0][1]
                last_activity = activities[-1][1]

                active_time = last_activity - first_activity
                away_time = total_time - active_time
                hourly_counts = {'core productive': 0, 'productive': 0, 'idle': 0, 'unproductive': 0}

                for page_title, _ in activities:
                    logging.info(f"Activity: {page_title}")
                    category, confidence, _ = predict_category(page_title)
                    status = map_category_to_productivity(category)
                    hourly_counts[status] += 1
                    logging.info(f"Activity: {page_title}, Prediction: {category}, Confidence: {confidence:.2f}")

                total_activities = sum(hourly_counts.values())
                if total_activities > 0:
                    core_productive_percentage = (hourly_counts['core productive'] / total_activities) * 100
                    productive_percentage = (hourly_counts['productive'] / total_activities) * 100
                    idle_percentage = (hourly_counts['idle'] / total_activities) * 100
                    unproductive_percentage = (hourly_counts['unproductive'] / total_activities) * 100
                else:
                    core_productive_percentage = 0
                    productive_percentage = 0
                    idle_percentage = 0
                    unproductive_percentage = 0
                away_percentage = (away_time / total_time) * 100

                logging.debug(f"Hour: {hour} - Core Productive: {core_productive_percentage}%, Productive: {productive_percentage}%, Idle: {idle_percentage}%, Unproductive: {unproductive_percentage}%, Away: {away_percentage}%")

                productivity_data.append([
                    {"productivity": "core productive", "percent": f"{core_productive_percentage:.2f}%"},
                    {"productivity": "productive", "percent": f"{productive_percentage:.2f}%"},
                    {"productivity": "idle", "percent": f"{idle_percentage:.2f}%"},
                    {"productivity": "unproductive", "percent": f"{unproductive_percentage:.2f}%"},
                    {"productivity": "away", "percent": f"{away_percentage:.2f}%"},
                    {"productivity": "", "percent": ""}
                ])

        logging.info("Productivity calculation completed.")
        return productivity_data
    except Exception as e:
        logging.error(f"Error calculating productivity: {e}")
        return []

def calculate_working_hours(activities):
    try:
        if not activities:
            return "0h 0m"
        first_activity = activities[0][1]
        last_activity = activities[-1][1]
        working_duration = last_activity - first_activity
        hours, remainder = divmod(working_duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        working_hours = f"{int(hours)}h {int(minutes)}m"
        return working_hours
    except Exception as e:
        logging.error(f"Error calculating working hours: {e}")
        return "0h 0m"
@app.route('/calculate_hourly_productivity', methods=['GET'])
def calculate_hourly_productivity():
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    organization_uid = request.args.get('organization_uid')
    if not organization_uid:
        return jsonify({"error": "organization_uid is required"}), 400

    devices = get_all_device_ids(organization_uid)
    all_productivity_data = []

    for device in devices:
        device_id = device['device_uid']
        user_name = device['user_name']
        activities = get_user_activities(device_id, date)
        logging.info(f"Activities found for device ID {device_id} on date {date}.")
        productivity_data = calculate_productivity_internal(activities) if activities else [
            [{"productivity": "away", "percent": "100%"} for _ in range(6)] for _ in range(9)
        ]
        working_hours = calculate_working_hours(activities) if activities else "0h 0m"

        all_productivity_data.append({
            'name': user_name,
            'workingHour': working_hours,
            'productivityRecord': productivity_data
        })

    logging.info("Hourly productivity calculation completed for all devices.")
    return jsonify(all_productivity_data)

# New route to test productivity prediction based on title
@app.route('/predict_productivity', methods=['POST'])
def predict_productivity_route():
    data = request.get_json()
    title = data.get('title', '')

    if not title:
        return jsonify({"productivity": "away", "percent": "100%"}), 200

    category, confidence, already_in_others = predict_category(title)

    if already_in_others:
        return jsonify({"category": "Others", "productivity": "idle", "percent": "100%", "message": "Please update its category"}), 200

    productivity = map_category_to_productivity(category)

    logging.info(f"Prediction accuracy for '{title}': {confidence * 100:.2f}%")

    return jsonify({"category": category, "productivity": productivity, "percent": "100%", "confidence": f"{confidence * 100:.2f}%"}), 200

# Route to get uncategorized titles and predefined categories
@app.route('/map_category', methods=['GET'])
def get_uncategorized_titles():
    others = category_list.get("Others", [])
    return jsonify({
        "others": others,
        "CategoryList": {i: cat for i, cat in enumerate(category_list.keys(), 1)}
    })

# Route to map a title to a category
@app.route('/map_category', methods=['POST'])
def map_category():
    data = request.get_json()
    otherCategoryCode = data.get('otherCategoryCode')
    categoryCode = data.get('categoryCode')

    try:
        otherCategoryCode = int(otherCategoryCode)
        categoryCode = int(categoryCode)
    except ValueError:
        return jsonify({"status": "error", "message": "Please provide valid numeric codes."}), 400

    if otherCategoryCode < 0 or otherCategoryCode >= len(category_list["Others"]):
        return jsonify({"status": "error", "message": "Please provide a valid 'otherCategoryCode' index."}), 400

    if categoryCode < 1 or categoryCode > len(category_list):
        return jsonify({"status": "error", "message": "Please provide a valid 'categoryCode' index."}), 400

    title_to_map = category_list["Others"].pop(otherCategoryCode)
    category_name = list(category_list.keys())[categoryCode - 1]

    if title_to_map and category_name:
        category_list[category_name].append(title_to_map)
        save_category_list()

    return jsonify({"status": "success", "message": "Category mapping updated."})

@app.route('/getUserProductivity/<device_id>', methods=['GET'])
def get_user_productivity(device_id):
    # Get the start date from the query parameters
    start_date_str = request.args.get('from')
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    except Exception as e:
        logging.error(f"Invalid 'from' date format: {e}")
        return jsonify({"error": "Invalid 'from' date format. Use YYYY-MM-DD."}), 400

    # Initialize the data format for response
    response_data = {
        "Core Productivity": [],
        "Productivity": [],
        "Unproductivity": [],
        "Idle": [],
        "Away": []
    }

    # Iterate over the 7-day range
    for day_offset in range(7):
        current_date = start_date + timedelta(days=day_offset)
        activities = get_user_activities(device_id, current_date.strftime('%Y-%m-%d'))

        if activities:
            hourly_counts = {'core productive': 0, 'productive': 0, 'idle': 0, 'unproductive': 0}
            total_activities = len(activities)

            for page_title, _ in activities:
                category, confidence, _ = predict_category(page_title)
                status = map_category_to_productivity(category)
                hourly_counts[status] += 1

            core_productive_percentage = (hourly_counts['core productive'] / total_activities) * 100
            productive_percentage = (hourly_counts['productive'] / total_activities) * 100
            unproductive_percentage = (hourly_counts['unproductive'] / total_activities) * 100
            Idle_percentage = (hourly_counts['idle'] / total_activities) * 100
        else:
            core_productive_percentage = 0
            productive_percentage = 0
            unproductive_percentage = 0
            Idle_percentage = 0

        away_percentage = 100 - (core_productive_percentage + productive_percentage + unproductive_percentage + Idle_percentage)

        # Append daily percentages to the response
        response_data["Core Productivity"].append(round(core_productive_percentage, 2))
        response_data["Productivity"].append(round(productive_percentage, 2))
        response_data["Unproductivity"].append(round(unproductive_percentage, 2))
        response_data["Idle"].append(round(Idle_percentage, 2))
        response_data["Away"].append(round(away_percentage, 2))

    logging.info(f"User productivity data prepared for device_id {device_id}")
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)

#  Only need to change the gap or widths of the data in teh frontend.