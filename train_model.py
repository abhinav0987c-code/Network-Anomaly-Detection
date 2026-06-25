import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
from sklearn.metrics import f1_score, classification_report

# Dataset URL
URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt"
DATA_FILE = "KDDTrain+_20Percent.txt"

def download_data():
    if not os.path.exists(DATA_FILE):
        print(f"Downloading {DATA_FILE}...")
        r = requests.get(URL)
        with open(DATA_FILE, 'wb') as f:
            f.write(r.content)
    else:
        print(f"{DATA_FILE} already exists.")

def train():
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 
        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
        'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
    ]

    df = pd.read_csv(DATA_FILE, header=None, names=columns)
    
    # Attack Mapping
    attack_map = {
        'normal': 'NORMAL',
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'mailbomb': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
        'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'named': 'R2L',
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'ps': 'U2R', 'rootkit': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
    }
    
    df['attack_category'] = df['label'].map(attack_map).fillna('NORMAL')
    
    # Select features requested by user + top influencers
    selected_features = ['protocol_type', 'src_bytes', 'dst_bytes', 'service', 'flag']
    # Expansion: Duration and Logged In
    expansion_features = ['duration', 'logged_in']
    # Help features for context
    context_features = ['count', 'srv_count', 'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count']
    all_features = selected_features + expansion_features + context_features

    X = df[all_features].copy()
    
    # Target Encoding
    target_le = LabelEncoder()
    y = target_le.fit_transform(df['attack_category'])
    print(f"Target Classes: {target_le.classes_}")

    # Encoding
    encoders = {}
    for col in ['protocol_type', 'service', 'flag']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(objective='multi:softprob', num_class=len(target_le.classes_), random_state=42)
    lgbm = LGBMClassifier(objective='multiclass', num_class=len(target_le.classes_), random_state=42)
    cat = CatBoostClassifier(loss_function='MultiClass', silent=True, random_state=42)

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
        voting='soft'
    )

    print("Training Ensemble Model (Multi-Class)...")
    ensemble.fit(X_train, y_train)
    
    accuracy = ensemble.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Calculate F1-Score
    y_pred = ensemble.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1-Score (Weighted): {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_le.classes_))

    # Save individual models
    for name, est in ensemble.named_estimators_.items():
        joblib.dump(est, f'{name}_model.joblib')
        print(f"Saved {name}_model.joblib")
    
    # Save legacy/full ensemble if needed (Optional, keeping separate as requested)
    # joblib.dump(ensemble, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(encoders, 'encoders.joblib')
    joblib.dump(all_features, 'feature_names.joblib')
    joblib.dump(target_le, 'target_encoder.joblib')
    
    print("Model and preprocessing objects saved.")

if __name__ == "__main__":
    download_data()
    train()
