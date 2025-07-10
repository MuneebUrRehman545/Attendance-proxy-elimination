# Attendance proxy elimination
 Face Recognition Attendance System (Anti-Proxy Enabled)

An AI-powered automated attendance system using **DeepFace**, **OpenCV**, and **Facenet**, designed to detect and recognize real faces from a live webcam and generate accurate attendance reports — eliminating proxy or fake attendance in offices, classrooms, and events.



 What Makes It Unique?

Unlike traditional attendance methods (manual or RFID cards), this system:
- Prevents **proxy attendance** by comparing real-time face embeddings with pre-registered datasets.
-  Detects each face **multiple times during class** (10 times with intervals), not just once.
-  Matches faces using **DeepFace with cosine similarity**, reducing false recognition.

This makes it suitable for:
- University classrooms
- Office entry attendance
- Examination invigilation
- Online presence tracking



 Features

-  Real-time face detection using webcam
-  Face recognition via **DeepFace (Facenet model)**
- CSV-based attendance report generation
-  Status tags: Present, Early Left, Late Comer, Absent
-  Final summary with attendance percentage
-  Email reports automatically (via Gmail)
-  Strong anti-spoofing structure using embeddings and multiple detection.
-  

 How It Works

1. Loads known faces from `dataset/` folder (each folder = 1 person).
2. Captures webcam feed 10 times, 3 seconds apart.
3. Detects and saves faces, then generates embeddings.
4. Compares with stored embeddings using cosine similarity.
5. Generates:
   - `attendance_master.csv` → Ongoing report
   - `final_summary.csv` → End-of-course report (optional)
6. Sends email with attached summary (if last lecture).


