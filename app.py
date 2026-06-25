from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
import joblib
import numpy as np
import pandas as pd
import json
import os
import threading
import time
from datetime import datetime
from sniffer import PacketSniffer

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# Global state management
sniffer = None
sniffer_thread = None
current_mode = None # 'Real' or 'Demo'

# Load model and preprocessing objects
models = {
    'RF': joblib.load('rf_model.joblib'),
    'XGB': joblib.load('xgb_model.joblib'),
    'LGBM': joblib.load('lgbm_model.joblib'),
    'CAT': joblib.load('cat_model.joblib')
}
scaler = joblib.load('scaler.joblib')
encoders = joblib.load('encoders.joblib')
features = joblib.load('feature_names.joblib')
target_encoder = joblib.load('target_encoder.joblib')

def get_remediation_suggestions(attack_type):
    suggestions = {
        'DoS': [
            "Implement rate limiting on edge firewalls to prevent resource exhaustion.",
            "Deploy a DDoS mitigation service (e.g., Cloudflare) to filter malicious traffic.",
            "Configure connection limits and timeouts on web servers."
        ],
        'Probe': [
            "Disable unnecessary network services and close unused ports.",
            "Enable 'stealth mode' on firewalls to drop scan packets without responding.",
            "Implement an Intrusion Detection System (IDS) to alert on scanning patterns."
        ],
        'R2L': [
            "Enforce complex password policies and mandatory Multi-Factor Authentication (MFA).",
            "Audit remote access logs for unusual login patterns or brute-force attempts.",
            "Restrict access to administrative interfaces using IP whitelisting."
        ],
        'U2R': [
            "Apply the latest security patches to the OS and critical applications.",
            "Implement strict 'Principle of Least Privilege' for all system users.",
            "Monitor system logs for unauthorized use of 'sudo' or privilege changes."
        ],
        'NORMAL': [
            "Traffic appears safe. Continue regular security monitoring.",
            "Maintain regular security updates and routine vulnerability assessments.",
            "Ensure logging systems are functioning for future audits."
        ]
    }
    return suggestions.get(attack_type, ["No specific suggestions available."])

HISTORY_FILE = 'history.json'

BULK_HISTORY_FILE = 'bulk_history.json'
LIVE_HISTORY_FILE = 'live_history.json'
DEMO_HISTORY_FILE = 'demo_history.json'
file_lock = threading.Lock()

def _load_json(path):
    with file_lock:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                print(f"Warning: Corrupted {path}. Resetting.")
                with open(path, 'w') as f:
                    json.dump([], f)
    return []

def _save_json(path, data):
    with file_lock:
        with open(path, 'w') as f:
            json.dump(data, f)

def load_history(): return _load_json(HISTORY_FILE)
def save_history(history): _save_json(HISTORY_FILE, history)

def load_bulk_history(): return _load_json(BULK_HISTORY_FILE)
def save_bulk_history(history): _save_json(BULK_HISTORY_FILE, history)

def load_live_history():
    data = _load_json(LIVE_HISTORY_FILE)
    # Migration check: if the first item is a packet (no "results" key), wrap it
    if data and isinstance(data, list) and len(data) > 0 and 'results' not in data[0]:
        print("Migrating legacy LIVE history to session-based format...")
        new_data = [{
            "timestamp": data[0].get('timestamp', 'Legacy'),
            "total": len(data),
            "attacks": sum(1 for x in data if x.get('result') != 'NORMAL'),
            "normals": sum(1 for x in data if x.get('result') == 'NORMAL'),
            "distribution": {"NORMAL": sum(1 for x in data if x.get('result') == 'NORMAL'), "DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0},
            "results": data
        }]
        # Recalculate distribution
        for x in data:
            res = x.get('result')
            if res in new_data[0]['distribution']: new_data[0]['distribution'][res] += 1
        save_live_history(new_data)
        return new_data
    return data

def save_live_history(history): _save_json(LIVE_HISTORY_FILE, history)

def load_demo_history():
    data = _load_json(DEMO_HISTORY_FILE)
    if data and isinstance(data, list) and len(data) > 0 and 'results' not in data[0]:
        print("Migrating legacy DEMO history to session-based format...")
        new_data = [{
            "timestamp": data[0].get('timestamp', 'Legacy'),
            "total": len(data),
            "attacks": sum(1 for x in data if x.get('result') != 'NORMAL'),
            "normals": sum(1 for x in data if x.get('result') == 'NORMAL'),
            "distribution": {"NORMAL": sum(1 for x in data if x.get('result') == 'NORMAL'), "DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0},
            "results": data
        }]
        for x in data:
            res = x.get('result')
            if res in new_data[0]['distribution']: new_data[0]['distribution'][res] += 1
        save_demo_history(new_data)
        return new_data
    return data

def save_demo_history(history): _save_json(DEMO_HISTORY_FILE, history)

def get_attack_stats(history_list):
    """Helper to calculate counts and percentages for charts."""
    counts = {'NORMAL': 0, 'DoS': 0, 'Probe': 0, 'R2L': 0, 'U2R': 0}
    for item in history_list:
        res = item.get('result', 'NORMAL')
        if res in counts:
            counts[res] += 1
    
    res_dict = {
        'labels': list(counts.keys()),
        'values': list(counts.values())
    }
    print(f"DEBUG: chart_data = {res_dict}")
    return res_dict

def perform_prediction(data):
    """Common logic for manual, bulk, and real-time detection."""
    try:
        # Preprocess
        input_df = pd.DataFrame([data])
        for col in ['protocol_type', 'service', 'flag']:
            if col in encoders:
                le = encoders[col]
                try:
                    input_df[col] = le.transform(input_df[col])
                except ValueError:
                    input_df[col] = le.transform([le.classes_[0]])[0] # Default

        # Scale
        input_scaled_raw = scaler.transform(input_df[features])
        input_scaled = pd.DataFrame(input_scaled_raw, columns=features)

        # Predict using manual ensemble (Soft Voting)
        all_probs = []
        votes = {}
        
        for name, m in models.items():
            # Convert to numpy to avoid feature name warnings
            X_input = input_scaled.values if hasattr(input_scaled, 'values') else input_scaled
            # Probabilities
            probs = m.predict_proba(X_input)[0]
            all_probs.append(probs)
            
            # Individual Vote
            v_idx = np.argmax(probs)
            v_label = str(target_encoder.inverse_transform([v_idx])[0])
            votes[name] = v_label

        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        prediction_idx = np.argmax(avg_probs)
        result_label = str(target_encoder.inverse_transform([prediction_idx])[0])
        confidence = float(np.max(avg_probs) * 100)

        # Top Feature Contributor
        diffs = np.abs(input_scaled.iloc[0])
        top_idx = np.argmax(diffs)
        top_feature = features[top_idx].replace('_', ' ').title()

        return {
            'result': result_label,
            'confidence': f"{confidence:.2f}%",
            'votes': votes,
            'top_feature': top_feature,
            'suggestions': get_remediation_suggestions(result_label),
            'color': "green" if result_label == "NORMAL" else "red"
        }
    except Exception as e:
        raise ValueError(f"Model prediction failed: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        try:
            # Extract features from form
            protocol = request.form.get('protocol_type')
            service = request.form.get('service')
            flag = request.form.get('flag')
            src_bytes = float(request.form.get('src_bytes', 0))
            dst_bytes = float(request.form.get('dst_bytes', 0))
            count = int(request.form.get('count', 1))
            srv_count = int(request.form.get('srv_count', 1))
            same_srv_rate = float(request.form.get('same_srv_rate', 1.0))
            duration = float(request.form.get('duration', 0))
            logged_in = 1 if request.form.get('logged_in') == 'on' else 0
            
            # Contextual features
            data = {
                'protocol_type': protocol,
                'src_bytes': src_bytes,
                'dst_bytes': dst_bytes,
                'service': service,
                'flag': flag,
                'duration': duration,
                'logged_in': logged_in,
                'count': count,
                'srv_count': srv_count,
                'same_srv_rate': same_srv_rate,
                'diff_srv_rate': 0.0,
                'dst_host_srv_count': 1
            }
            
            # Use utility for prediction
            result_obj = perform_prediction(data)
            
            # Save to history
            history = load_history()
            history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'protocol': str(protocol),
                'service': str(service),
                'src_bytes': float(src_bytes),
                'dst_bytes': float(dst_bytes),
                'result': result_obj['result'],
                'confidence': result_obj['confidence'],
                'features': {k: (v.item() if hasattr(v, 'item') else v) for k, v in data.items()},
                'suggestions': [str(s) for s in result_obj['suggestions']]
            })
            save_history(history)
            
            return render_template('result.html', 
                                 result=result_obj['result'], 
                                 confidence=result_obj['confidence'], 
                                 color=result_obj['color'],
                                 votes=result_obj['votes'],
                                 top_feature=result_obj['top_feature'],
                                 suggestions=result_obj['suggestions'])
        except Exception as e:
            return f"Error processing input: {str(e)}", 400
            
    return render_template('detect.html', 
                          protocols=list(encoders['protocol_type'].classes_),
                          services=list(encoders['service'].classes_[:15]), # Limit for UI
                          flags=list(encoders['flag'].classes_))

@app.route('/live')
def live_monitor():
    return render_template('live.html',
                          protocols=list(encoders['protocol_type'].classes_),
                          services=list(encoders['service'].classes_[:15]),
                          flags=list(encoders['flag'].classes_))

@app.route('/api/live_detect', methods=['POST'])
def api_live_detect():
    try:
        data = request.json
        # Handle form-to-JSON mapping if needed but usually just pass it
        result_obj = perform_prediction(data)
        
        # Save to history silently
        history = load_history()
        history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'protocol': str(data.get('protocol_type', 'TCP')),
            'service': str(data.get('service', 'HTTP')),
            'src_bytes': float(data.get('src_bytes', 0)),
            'dst_bytes': float(data.get('dst_bytes', 0)),
            'result': result_obj['result'],
            'confidence': result_obj['confidence'],
            'features': data,
            'suggestions': result_obj['suggestions']
        })
        save_history(history)
        
        return jsonify(result_obj)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# --- Socket.IO Real-Time Handlers ---

def sniffer_callback(data, is_real=True):
    """Callback for handle predictions for live packets."""
    try:
        if 'error' in data:
            socketio.emit('status', {'msg': f"Sniffing Error: {data['error']}", 'active': False})
            return

        # Perform prediction using existing logic
        result_obj = perform_prediction(data)
        
        # Add metadata
        result_obj['timestamp'] = datetime.now().strftime("%H:%M:%S")
        result_obj['protocol'] = data.get('protocol_type', 'TCP').upper()
        result_obj['service'] = data.get('service', 'HTTP').upper()
        result_obj['src_bytes'] = data.get('src_bytes', 0)
        result_obj['source'] = "Network" if is_real else "Dataset"
        result_obj['src_ip'] = data.get('src_ip', '127.0.0.1')
        result_obj['dst_ip'] = data.get('dst_ip', '127.0.0.1')
        result_obj['duration'] = data.get('duration', 0)
        
        # Prepare entry
        new_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'protocol': str(data.get('protocol_type', 'TCP')),
            'service': str(data.get('service', 'HTTP')),
            'src_bytes': float(data.get('src_bytes', 0)),
            'dst_bytes': float(data.get('dst_bytes', 0)),
            'src_ip': str(result_obj['src_ip']),
            'dst_ip': str(result_obj['dst_ip']),
            'result': result_obj['result'],
            'confidence': result_obj['confidence'],
            'features': {k: (v.item() if hasattr(v, 'item') else v) for k, v in data.items()},
            'suggestions': result_obj.get('suggestions', [])
        }
        
        # Save to active session in correct history file
        if is_real:
            history = load_live_history()
            if history:
                session = history[-1]
                session['results'].append(new_entry)
                session['total'] += 1
                # Standardize keys to match AI results
                res_key = new_entry['result']
                if res_key == 'NORMAL': 
                    session['normals'] += 1
                else: 
                    session['attacks'] += 1
                
                if 'distribution' not in session:
                    session['distribution'] = {"NORMAL": 0, "DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0}
                
                session['distribution'][res_key] = session['distribution'].get(res_key, 0) + 1
                save_live_history(history)
        else:
            history = load_demo_history()
            if history:
                session = history[-1]
                session['results'].append(new_entry)
                session['total'] += 1
                res_key = new_entry['result']
                if res_key == 'NORMAL': 
                    session['normals'] += 1
                else: 
                    session['attacks'] += 1
                
                if 'distribution' not in session:
                    session['distribution'] = {"NORMAL": 0, "DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0}
                
                session['distribution'][res_key] = session['distribution'].get(res_key, 0) + 1
                save_demo_history(history)
        
        # Emit to dashboard
        socketio.emit('new_detection', result_obj)
        
    except Exception as e:
        print(f"Callback error: {e}")

@socketio.on('toggle_monitor')
def handle_toggle(message):
    global sniffer, sniffer_thread, current_mode
    action = message.get('action') # 'start_real', 'start_demo', 'stop'
    
    if action == 'stop':
        if sniffer:
            sniffer.stop()
        current_mode = None
        emit('status', {'msg': 'Monitoring stopped', 'active': False})
        return

    if sniffer and sniffer.running:
        sniffer.stop()
        time.sleep(0.5)
        current_mode = None

    sniffer = PacketSniffer(callback=sniffer_callback, dataset_path="KDDTrain+_20Percent.txt")
    
    if action == 'start_real':
        if not sniffer.can_sniff_real:
            emit('status', {'msg': 'Error: No L3 socket available. Install Npcap/WinPcap for real sniffing.', 'active': False})
            return
        
        # Guard: Only create session if not already in real mode
        if current_mode != 'Real':
            current_mode = 'Real'
            history = load_live_history()
            history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total": 0, "attacks": 0, "normals": 0,
                "distribution": {"NORMAL": 0, "DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0},
                "results": []
            })
            if len(history) > 50: history.pop(0)
            save_live_history(history)

        sniffer_thread = threading.Thread(target=sniffer.start_sniffing)
        sniffer_thread.daemon = True
        sniffer_thread.start()
        emit('status', {'msg': 'Real-time sniffing started', 'active': True, 'mode': 'Real'})
    
    elif action == 'start_demo':
        # Guard: Only create session if not already in demo mode
        if current_mode != 'Demo':
            current_mode = 'Demo'
            history = load_demo_history()
            history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total": 0, "attacks": 0, "normals": 0,
                "distribution": {"NORMAL": 0, "DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0},
                "results": []
            })
            if len(history) > 50: history.pop(0)
            save_demo_history(history)

        sniffer_thread = threading.Thread(target=sniffer.start_demo)
        sniffer_thread.daemon = True
        sniffer_thread.start()
        emit('status', {'msg': 'Demo streaming started', 'active': True, 'mode': 'Demo'})

@app.route('/live_history')
def live_history():
    data = load_live_history()
    return render_template('live_history.html', history=list(enumerate(data))[::-1], mode="Live")

@app.route('/demo_history')
def demo_history():
    data = load_demo_history()
    return render_template('demo_history.html', history=list(enumerate(data))[::-1], mode="Demo")

@app.route('/live_session/<int:id>')
def live_session(id):
    data = load_live_history()
    if 0 <= id < len(data):
        session = data[id]
        chart_data = {
            'labels': list(session['distribution'].keys()),
            'values': list(session['distribution'].values())
        }
        return render_template('session_details.html', session=session, id=id, mode="Live", chart_data=chart_data)
    return "Session Not Found", 404

@app.route('/demo_session/<int:id>')
def demo_session(id):
    data = load_demo_history()
    if 0 <= id < len(data):
        session = data[id]
        chart_data = {
            'labels': list(session['distribution'].keys()),
            'values': list(session['distribution'].values())
        }
        return render_template('session_details.html', session=session, id=id, mode="Demo", chart_data=chart_data)
    return "Session Not Found", 404

@app.route('/delete_live/<int:index>')
def delete_live(index):
    history = load_live_history()
    if 0 <= index < len(history):
        history.pop(index)
        save_live_history(history)
    return redirect(url_for('live_history'))

@app.route('/delete_demo/<int:index>')
def delete_demo(index):
    history = load_demo_history()
    if 0 <= index < len(history):
        history.pop(index)
        save_demo_history(history)
    return redirect(url_for('demo_history'))

@app.route('/clear_live_history')
def clear_live_history():
    save_live_history([])
    return redirect(url_for('live_history'))

@app.route('/clear_demo_history')
def clear_demo_history():
    save_demo_history([])
    return redirect(url_for('demo_history'))

@app.route('/dashboard')
def dashboard():
    history = load_history()
    
    # Simple stats
    stats = {
        'total': len(history),
        'NORMAL': len([h for h in history if h['result'] == 'NORMAL']),
        'DoS': len([h for h in history if h['result'] == 'DoS']),
        'Probe': len([h for h in history if h['result'] == 'Probe']),
        'R2L': len([h for h in history if h['result'] == 'R2L']),
        'U2R': len([h for h in history if h['result'] == 'U2R'])
    }
    
    # Chart Data 1: Attack Distribution
    attack_counts = {}
    for h in history:
        res = h['result']
        attack_counts[res] = attack_counts.get(res, 0) + 1
    
    # Chart Data 2: Protocol Analysis
    proto_counts = {}
    for h in history:
        p = h.get('protocol', 'UNKNOWN').upper()
        proto_counts[p] = proto_counts.get(p, 0) + 1
    
    chart_data = {
        'chart_labels': list(attack_counts.keys()),
        'chart_values': list(attack_counts.values()),
        'proto_labels': list(proto_counts.keys()),
        'proto_values': list(proto_counts.values())
    }
    
    return render_template('dashboard.html', 
                          history=list(enumerate(history))[::-1], 
                          bulk_history=list(enumerate(load_bulk_history()))[::-1],
                          stats=stats, 
                          chart_data=chart_data)

@app.route('/history_details/<int:id>')
def history_details(id):
    history = load_history()
    if 0 <= id < len(history):
        entry = history[id]
        # Ensure suggestions exist for legacy history items if any
        if 'suggestions' not in entry:
            entry['suggestions'] = get_remediation_suggestions(entry['result'])
        return render_template('history_details.html', entry=entry, id=id)
    return redirect(url_for('dashboard'))

@app.route('/delete/<int:index>')
def delete_entry(index):
    history = load_history()
    if 0 <= index < len(history):
        history.pop(index)
        save_history(history)
    return redirect(url_for('dashboard'))

@app.route('/clear_history')
def clear_history():
    save_history([])
    save_bulk_history([])
    return redirect(url_for('dashboard'))

@app.route('/bulk_details/<int:id>')
def bulk_details(id):
    bulk_history = load_bulk_history()
    if 0 <= id < len(bulk_history):
        file_data = bulk_history[id]
        chart_data = get_attack_stats(file_data.get('results', []))
        return render_template('bulk_details.html', file_data=file_data, chart_data=chart_data)
    return redirect(url_for('dashboard'))

@app.route('/delete_bulk/<int:id>')
def delete_bulk(id):
    bulk_history = load_bulk_history()
    if 0 <= id < len(bulk_history):
        bulk_history.pop(id)
        save_bulk_history(bulk_history)
    return redirect(url_for('dashboard'))

@app.route('/delete_session/<mode>/<int:session_id>')
def delete_session(mode, session_id):
    if mode == 'live':
        history = load_live_history()
        if 0 <= session_id < len(history):
            history.pop(session_id)
            save_live_history(history)
        return redirect(url_for('live_history'))
    elif mode == 'demo':
        history = load_demo_history()
        if 0 <= session_id < len(history):
            history.pop(session_id)
            save_demo_history(history)
        return redirect(url_for('demo_history'))
    return redirect(url_for('home'))

@app.route('/batch', methods=['GET', 'POST'])
def batch_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        
        if file and file.filename.endswith('.csv'):
            try:
                # Load the data
                import pandas as pd # Already at top but for safety in this scope if needed
                df = pd.read_csv(file)
                
                # Check for required columns and add defaults for missing ones
                features = joblib.load('feature_names.joblib')
                for f in features:
                    if f not in df.columns:
                        if f in ['diff_srv_rate', 'dst_host_srv_count']:
                            df[f] = 0.0 if f == 'diff_srv_rate' else 1
                        else:
                            return f"Missing required column: {f}", 400
                
                # Preprocess (Encoding)
                process_df = df.copy()
                for col in ['protocol_type', 'service', 'flag']:
                    le = encoders[col]
                    process_df[col] = process_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0])
                
                # Predict using chosen model (Random Forest as default for batch)
                X = scaler.transform(process_df[features])
                preds_idx = models['RF'].predict(X)
                preds = target_encoder.inverse_transform(preds_idx)
                
                # Summarize
                total = len(preds)
                normals = int(np.sum(preds == 'NORMAL'))
                attacks_dist = {
                    'DoS': int(np.sum(preds == 'DoS')),
                    'Probe': int(np.sum(preds == 'Probe')),
                    'R2L': int(np.sum(preds == 'R2L')),
                    'U2R': int(np.sum(preds == 'U2R'))
                }
                total_attacks = total - normals
                
                # Save results to Grouped Bulk History
                bulk_history = load_bulk_history()
                file_results = []
                for i in range(total):
                    res_label = str(preds[i])
                    # Capture basic features for the detail modal
                    feat_dict = {
                        'protocol_type': str(df.iloc[i]['protocol_type']),
                        'service': str(df.iloc[i]['service']),
                        'flag': str(df.iloc[i]['flag']),
                        'src_bytes': float(df.iloc[i]['src_bytes']),
                        'dst_bytes': float(df.iloc[i]['dst_bytes']),
                        'duration': float(df.iloc[i].get('duration', 0))
                    }
                    file_results.append({
                        'protocol': feat_dict['protocol_type'],
                        'service': feat_dict['service'],
                        'src_bytes': feat_dict['src_bytes'],
                        'dst_bytes': feat_dict['dst_bytes'],
                        'result': res_label,
                        'suggestions': get_remediation_suggestions(res_label),
                        'features': feat_dict
                    })
                
                bulk_history.append({
                    'filename': file.filename,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'total': total,
                    'attacks': total_attacks,
                    'normals': normals,
                    'distribution': attacks_dist,
                    'results': file_results
                })
                save_bulk_history(bulk_history)
                
                return render_template('batch_result.html', 
                                     total=total, 
                                     attacks=total_attacks, 
                                     normals=normals,
                                     distribution=attacks_dist)
            except Exception as e:
                return f"Error processing CSV: {str(e)}", 400
    
    return render_template('batch.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5005, use_reloader=False, allow_unsafe_werkzeug=True)
