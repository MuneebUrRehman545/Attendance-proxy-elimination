import cv2
import os
import time
from datetime import datetime
from deepface import DeepFace
from scipy.spatial.distance import cosine
import pandas as pd
import requests
import smtplib
from email.message import EmailMessage

# ------------------------- CONFIG -------------------------
model_name = "Facenet"
threshold = 0.4
dataset_path = r"F:\\4 SEMESTER\\PROG FOR AI\\Project\\dataset"
output_path = "final_attendance"
os.makedirs(output_path, exist_ok=True)
temp_image_dir = "captures"
os.makedirs(temp_image_dir, exist_ok=True)

# Gmail configuration
sender_email = "....@gmail.com"
receiver_email = ".....@gmail.com"
app_password = "your email app password"

# ------------------------- Load Known Faces -------------------------
def build_dataset_embeddings(dataset_path):
    embeddings = {}
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            try:
                embedding = DeepFace.represent(img_path=img_path, model_name=model_name)[0]["embedding"]
                embeddings[person] = embedding
                print(f"Loaded embedding for: {person}")
                break
            except Exception as e:
                print(f" Error loading {img_path}: {e}")
    return embeddings

# ------------------------- Recognize One Face -------------------------
def recognize_face(face_img, known_embeddings):
    try:
        result = DeepFace.represent(img_path=face_img, model_name=model_name)[0]
        captured_embedding = result["embedding"]
    except Exception as e:
        print(f"? Failed to generate embedding for captured face: {e}")
        return "Unknown"

    min_dist = float("inf")
    identity = "Unknown"
    for name, ref_embedding in known_embeddings.items():
        dist = cosine(captured_embedding, ref_embedding)
        if dist < threshold and dist < min_dist:
            min_dist = dist
            identity = name
    return identity

# ------------------------- Capture & Detect -------------------------
def capture_and_detect(cam, known_embeddings, capture_num):
    ret, frame = cam.read()
    if not ret:
        print("? Failed to capture frame.")
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    detected_names = []

    if len(faces) == 0:
        print(" No faces detected in this frame.")
        return []

    for i, (x, y, w, h) in enumerate(faces):
        face_crop = frame[y:y+h, x:x+w]
        temp_path = os.path.join(temp_image_dir, f"capture_{capture_num}_face_{i}.jpg")
        try:
            cv2.imwrite(temp_path, face_crop)
            print(f" Saved face to {temp_path}")
            name = recognize_face(temp_path, known_embeddings)
            print(f"Recognized: {name}")
            if name != "Unknown":
                detected_names.append(name)
        except Exception as e:
            print(f" Error saving or recognizing face: {e}")

    return detected_names

# ------------------------- Generate Attendance Report -------------------------
def generate_final_attendance(all_detections, total_captures):
    all_students = set()
    for detection in all_detections:
        all_students.update(detection)

    attendance_result = {}
    for student in all_students:
        detections = [i for i, capture in enumerate(all_detections) if student in capture]
        count = len(detections)

        if count >= 6:
            status = "Present"
        elif 0 < count < 5:
            status = "Early Left"
        elif all(i >= 2 for i in detections):
            status = "Late Comer"
        else:
            status = "Unknown"

        attendance_result[student] = status

    return attendance_result

# ------------------------- Update Master CSV -------------------------
def update_master_csv(attendance_result, dataset_path, master_csv_path):
    registered_students = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    time_col = datetime.now().strftime("%Y-%m-%d %H:%M")

    today_data = {student: attendance_result.get(student, "Absent") for student in registered_students}
    total_students = len(registered_students)
    present = list(today_data.values()).count("Present")
    early = list(today_data.values()).count("Early Left")
    late = list(today_data.values()).count("Late Comer")
    absent = list(today_data.values()).count("Absent")

    df = pd.DataFrame.from_dict(today_data, orient='index', columns=[time_col])
    df.index.name = 'Name'

    if os.path.exists(master_csv_path):
        existing_df = pd.read_csv(master_csv_path)
        existing_df.set_index('Name', inplace=True)
        updated_df = existing_df.join(df, how='outer')
    else:
        updated_df = df

    # Append summary rows
    updated_df.loc['--Total Students--'] = [total_students] + [None]*(updated_df.shape[1]-1)
    updated_df.loc['--Present--'] = [present] + [None]*(updated_df.shape[1]-1)
    updated_df.loc['--Early Left--'] = [early] + [None]*(updated_df.shape[1]-1)
    updated_df.loc['--Late Comer--'] = [late] + [None]*(updated_df.shape[1]-1)
    updated_df.loc['--Absent--'] = [absent] + [None]*(updated_df.shape[1]-1)

    updated_df.to_csv(master_csv_path)
    print(f" Attendance saved to {master_csv_path}")

    return updated_df

# ------------------------- Final Summary -------------------------
def generate_final_summary(df, summary_path):
    attendance_df = df[df.index.str.startswith("--") == False]
    total_lectures = len(df.columns) - 1  # exclude Name
    attendance_percent = attendance_df.apply(lambda row: (row == "Present").sum() / total_lectures * 100 if total_lectures > 0 else 0, axis=1)
    attendance_df['Attendance %'] = attendance_percent.round(2)
    attendance_df.to_csv(summary_path)
    print(f"Final summary saved to {summary_path}")
    return summary_path

# ------------------------- Send via Gmail -------------------------
def send_email_with_csv(sender_email, receiver_email, app_password, file_path):
    msg = EmailMessage()
    msg['Subject'] = 'Final Attendance Summary'
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content('Attached is the final attendance summary.')

    with open(file_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='application', subtype='octet-stream', filename=os.path.basename(file_path))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, app_password)
        smtp.send_message(msg)

    print(" Email sent to:", receiver_email)

# ------------------------- Main Execution -------------------------
print(" Building known face embeddings...")
known_embeddings = build_dataset_embeddings(dataset_path)
print(" Embeddings loaded.\n")

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

print(" Starting simulated class monitoring (10 captures, 3 sec apart)...\n")
total_captures = 10
capture_interval = 3

all_detections = []
for i in range(total_captures):
    print(f"\n Capture {i+1}/{total_captures}")
    names = capture_and_detect(cam, known_embeddings, i)
    print(f" Detected in Capture {i+1}: {names}")
    all_detections.append(names)
    time.sleep(capture_interval)

cam.release()
cv2.destroyAllWindows()

# ------------------------- Final Attendance -------------------------
try:
    attendance_result = generate_final_attendance(all_detections, total_captures)
    master_csv_path = os.path.join(output_path, "attendance_master.csv")
    updated_df = update_master_csv(attendance_result, dataset_path, master_csv_path)

    response = input("\n Is this the last lecture? (yes/no): ").strip().lower()
    if response == 'yes':
        summary_path = os.path.join(output_path, "final_summary.csv")
        final_summary_path = generate_final_summary(updated_df, summary_path)
        send_email_with_csv(sender_email, receiver_email, app_password, final_summary_path)

except Exception as e:
    print("? Error during attendance processing:", e)

print(" Script execution completed.")
