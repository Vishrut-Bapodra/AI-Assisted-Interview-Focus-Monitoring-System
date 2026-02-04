# AI-Driven Interview Focus Monitoring System

An AI-powered computer vision system designed to monitor candidate focus during online interviews using real-time webcam analysis. The system evaluates head orientation, eye visibility, and facial alignment over time to determine whether a candidate is actively focused on the interview screen.

This project aims to provide an objective, automated focus assessment mechanism that can assist interviewers during remote interviews.

---

## 🚀 Features

- Real-time webcam-based monitoring
- Facial landmark and iris tracking
- Head pose and nose-position deviation detection
- Temporal focus validation to avoid false positives
- Automatic warning generation on sustained distraction
- Session-based logging with screenshots and event data
- Lightweight and local execution (no cloud dependency)

---

## 🧠 How It Works

1. Captures live video feed using the system webcam.
2. Detects facial landmarks and iris positions per frame.
3. Tracks nose movement, head alignment, and eye visibility.
4. Applies temporal logic to verify sustained attention loss.
5. Logs focus events, warnings, and screenshots per session.
6. Outputs structured session data for later analysis.

---

## 📂 Project Structure
    
    AI-Driven-Interview-Focus-Monitoring-System/
    │
    ├── app.py 
    ├── vision.py 
    ├── temporal.py 
    ├── reasoning.py 
    ├── logger.py 
    └──  requirements.txt

---

## 🛠️ Tech Stack & Libraries

- Python
- OpenCV
- MediaPipe Face Mesh
- NumPy
- Math & Time (standard library)

---

## ⚙️ Installation & Setup

1. Clone the repository
    ```bash
    git clone <repo-url>
    cd rag-model

2. Navigate to the project directory
    ```bash
    cd AI-Driven-Interview-Focus-Monitoring-System

3. Install dependencies
    ```bash
    pip install -r requirements.txt

4. Run the application
     ```bash
     python app.py
