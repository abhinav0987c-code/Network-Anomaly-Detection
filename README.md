AI-Powered Network Anomaly Detection System
An advanced, real-time network traffic analysis and anomaly detection system. This application monitors network traffic (either live via packet sniffing or simulated from the NSL-KDD dataset) and uses a machine learning ensemble model to classify traffic and detect potential network intrusion attempts.

Key Features
Ensemble Classification: Employs a soft-voting ensemble of state-of-the-art machine learning models:
Random Forest (RF)
XGBoost (XGB)
LightGBM (LGBM)
CatBoost (CAT)
Intrusion Categories Detected:
DoS (Denial of Service)
Probe (Network surveillance/scanning)
R2L (Remote-to-Local unauthorized access)
U2R (User-to-Root privilege escalation)
NORMAL (Legitimate traffic)
Real-Time Dashboard: An interactive web interface using Flask and Flask-SocketIO for real-time traffic statistics, streaming detection, and historical analytics.
Dual Mode Sniffing:
Real Mode: Live packet sniffing on loopback or network interfaces using Scapy.
Demo Mode: Simulated stream reading directly from the KDDTrain+_20Percent.txt dataset.
Bulk Analysis: Upload CSV files of network packets for bulk classification and analysis.
Remediation Guidance: Automatically generates actionable security suggestions and mitigation advice based on the detected attack type.
Installation & Setup
1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

Windows Packet Sniffing Requirement
To capture real network traffic on Windows, you must install Npcap or WinPcap:

Download and install Npcap.
During installation, make sure to enable the "Install Npcap in WinPcap API-compatible Mode" option.
2. Clone the Repository
git clone <your-github-repo-url>
cd network-anomaly-detection-Project
3. Install Dependencies
Install all required libraries using pip:

pip install flask flask-socketio joblib numpy pandas xgboost lightgbm catboost scapy
Running the Application
Start the Flask application server:

python app.py
Open your browser and navigate to:

http://127.0.0.1:5005
Project Structure
├── app.py                       # Main Flask web application and Socket.IO handler
├── sniffer.py                   # Real-time packet sniffer and feature extractor (Scapy)
├── train_model.py               # ML training script to generate the models
├── KDDTrain+_20Percent.txt      # NSL-KDD dataset used for training and demo mode
├── static/                      # Frontend static assets (CSS, JS, Images)
├── templates/                   # HTML templates (Dashboard, Live view, etc.)
├── *.joblib                     # Saved preprocessors, encoders, and ML models
└── README.md                    # Project documentation
How It Works
Packet Capture: The packet sniffer (sniffer.py) captures IP packets via Scapy, parsing protocols (TCP, UDP, ICMP) and extracting connection features (e.g., source/destination bytes, flags, protocol type, rolling window counts).
Feature Preprocessing: Extracted features are passed through the saved encoders (encoders.joblib, target_encoder.joblib) and scaled via scaler.joblib.
Ensemble Prediction: The processed feature vector is sent to the four trained ML models. Their probability scores are aggregated (soft voting) to output the final prediction label and its confidence percentage.
UI Dashboard Update: The classification result is pushed to the client via Socket.IO, updating the real-time graphs and triggering remediation suggestions if an anomaly is found.
Requirements
Python 3.8+, Flask, Flask-SocketIO, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost, Scapy, Npcap / WinPcap (required for live packet capture on Windows).
