#!/usr/bin/env python3
"""
0XVENOM - Network Intrusion Detection System (NIDS)
Detects malicious and unauthorized access in computer networks using Machine Learning.
Author: Adarsh Kumar Singh

Enhanced Features:
- Model Saving/Loading
- Web Interface (Streamlit)
- Real-time Packet Capture
- Live Traffic Classification
"""

import os
import sys
import argparse
import time
import json
import joblib
import threading
import queue
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import warnings
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import RFE
import itertools
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from tabulate import tabulate
import colorama
from colorama import Fore, Back, Style

# Network packet capture imports
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: Scapy not available. Real-time packet capture disabled.")

# Initialize colorama for cross-platform colored output
colorama.init()

class PacketCapture:
    """Real-time packet capture and feature extraction"""
    
    def __init__(self, interface=None):
        self.interface = interface
        self.packet_queue = queue.Queue()
        self.is_capturing = False
        self.capture_thread = None
        
    def extract_features(self, packet):
        """Extract features from network packet"""
        features = {
            'duration': 0,
            'protocol_type': 0,
            'service': 0,
            'flag': 0,
            'src_bytes': 0,
            'dst_bytes': 0,
            'land': 0,
            'wrong_fragment': 0,
            'urgent': 0,
            'hot': 0,
            'num_failed_logins': 0,
            'logged_in': 0,
            'num_compromised': 0,
            'root_shell': 0,
            'su_attempted': 0,
            'num_root': 0,
            'num_file_creations': 0,
            'num_shells': 0,
            'num_access_files': 0,
            'num_outbound_cmds': 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            'count': 1,
            'srv_count': 1,
            'serror_rate': 0,
            'srv_serror_rate': 0,
            'rerror_rate': 0,
            'srv_rerror_rate': 0,
            'same_srv_rate': 1,
            'diff_srv_rate': 0,
            'srv_diff_host_rate': 0,
            'dst_host_count': 1,
            'dst_host_srv_count': 1,
            'dst_host_same_srv_rate': 1,
            'dst_host_diff_srv_rate': 0,
            'dst_host_same_src_port_rate': 0,
            'dst_host_srv_diff_host_rate': 0,
            'dst_host_serror_rate': 0,
            'dst_host_srv_serror_rate': 0,
            'dst_host_rerror_rate': 0,
            'dst_host_srv_rerror_rate': 0
        }
        
        if packet.haslayer(IP):
            ip_layer = packet[IP]
            
            # Protocol type
            if packet.haslayer(TCP):
                features['protocol_type'] = 1  # TCP
                tcp_layer = packet[TCP]
                features['src_bytes'] = len(packet)
                features['dst_bytes'] = len(packet)
                
                # Service port mapping (simplified)
                dst_port = tcp_layer.dport
                if dst_port == 80:
                    features['service'] = 1  # HTTP
                elif dst_port == 443:
                    features['service'] = 2  # HTTPS
                elif dst_port == 21:
                    features['service'] = 3  # FTP
                elif dst_port == 22:
                    features['service'] = 4  # SSH
                elif dst_port == 23:
                    features['service'] = 5  # Telnet
                
                # TCP flags
                if tcp_layer.flags & 0x02:  # SYN
                    features['flag'] = 1
                elif tcp_layer.flags & 0x10:  # ACK
                    features['flag'] = 2
                elif tcp_layer.flags & 0x01:  # FIN
                    features['flag'] = 3
                    
            elif packet.haslayer(UDP):
                features['protocol_type'] = 2  # UDP
                features['src_bytes'] = len(packet)
                features['dst_bytes'] = len(packet)
                
            elif packet.haslayer(ICMP):
                features['protocol_type'] = 3  # ICMP
                features['src_bytes'] = len(packet)
                
        return features
        
    def packet_handler(self, packet):
        """Handle captured packets"""
        try:
            features = self.extract_features(packet)
            timestamp = datetime.now()
            
            packet_data = {
                'timestamp': timestamp,
                'features': features,
                'raw_packet': str(packet.summary()) if hasattr(packet, 'summary') else str(packet)
            }
            
            self.packet_queue.put(packet_data)
            
        except Exception as e:
            print(f"Error processing packet: {e}")
            
    def start_capture(self, packet_count=0):
        """Start packet capture"""
        if not SCAPY_AVAILABLE:
            print("Scapy not available. Cannot start packet capture.")
            return False
            
        self.is_capturing = True
        
        def capture_worker():
            try:
                sniff(
                    iface=self.interface,
                    prn=self.packet_handler,
                    count=packet_count,
                    stop_filter=lambda x: not self.is_capturing
                )
            except Exception as e:
                print(f"Capture error: {e}")
                
        self.capture_thread = threading.Thread(target=capture_worker)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        return True
        
    def stop_capture(self):
        """Stop packet capture"""
        self.is_capturing = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
            
    def get_packets(self):
        """Get captured packets from queue"""
        packets = []
        while not self.packet_queue.empty():
            try:
                packets.append(self.packet_queue.get_nowait())
            except queue.Empty:
                break
        return packets

class ModelManager:
    """Handle model saving, loading, and management"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
    def save_model(self, model, model_name, scaler=None, features=None, metadata=None):
        """Save trained model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.joblib"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # Prepare model package
        model_package = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'metadata': metadata or {},
            'timestamp': timestamp,
            'model_name': model_name
        }
        
        # Save model
        joblib.dump(model_package, model_path)
        
        # Save metadata separately
        metadata_path = os.path.join(self.models_dir, f"{model_name}_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                'model_name': model_name,
                'timestamp': timestamp,
                'features': features,
                'metadata': metadata or {}
            }, f, indent=2)
            
        return model_path
        
    def load_model(self, model_path):
        """Load saved model"""
        try:
            model_package = joblib.load(model_path)
            return model_package
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
            
    def list_models(self):
        """List all saved models"""
        models = []
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.joblib'):
                model_path = os.path.join(self.models_dir, filename)
                try:
                    package = joblib.load(model_path)
                    models.append({
                        'filename': filename,
                        'path': model_path,
                        'name': package.get('model_name', 'Unknown'),
                        'timestamp': package.get('timestamp', 'Unknown'),
                        'features': len(package.get('features', [])) if package.get('features') else 0
                    })
                except:
                    continue
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)

class VenomNIDS:
    def __init__(self):
        self.models = {}
        self.selected_features = []
        self.scaler = StandardScaler()
        self.model_manager = ModelManager()
        self.packet_capture = PacketCapture()
        self.X_train = None
        self.Y_train = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.best_model_name = None
        
    def print_banner(self):
        """Display the 0XVENOM banner"""
        banner = f"""
{Fore.RED}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  {Fore.GREEN}  ██████╗ ██╗  ██╗██╗   ██╗███████╗███╗   ██╗ ██████╗ ███╗   ███╗        {Fore.RED}║
║  {Fore.GREEN} ██╔═████╗╚██╗██╔╝██║   ██║██╔════╝████╗  ██║██╔═══██╗████╗ ████║        {Fore.RED}║
║  {Fore.GREEN} ██║██╔██║ ╚███╔╝ ██║   ██║█████╗  ██╔██╗ ██║██║   ██║██╔████╔██║        {Fore.RED}║
║  {Fore.GREEN} ████╔╝██║ ██╔██╗ ╚██╗ ██╔╝██╔══╝  ██║╚██╗██║██║   ██║██║╚██╔╝██║        {Fore.RED}║
║  {Fore.GREEN} ╚██████╔╝██╔╝ ██╗ ╚████╔╝ ███████╗██║ ╚████║╚██████╔╝██║ ╚═╝ ██║        {Fore.RED}║
║  {Fore.GREEN}  ╚═════╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝        {Fore.RED}║
║                                                                              ║
║  {Fore.CYAN}            Network Intrusion Detection System (NIDS)                    {Fore.RED}║
║  {Fore.YELLOW}        Enhanced with Real-time Monitoring & Web Interface             {Fore.RED}║
║  {Fore.MAGENTA}                    Author: Adarsh Kumar Singh                          {Fore.RED}║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
        """
        print(banner)
        
    def print_status(self, message, status="INFO"):
        """Print colored status messages"""
        colors = {
            "INFO": Fore.CYAN,
            "SUCCESS": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "PROCESSING": Fore.MAGENTA
        }
        print(f"{colors.get(status, Fore.WHITE)}[{status}] {message}{Style.RESET_ALL}")
        
    def load_data(self, train_path, test_path):
        """Load training and testing datasets"""
        try:
            self.print_status("Loading training data...", "PROCESSING")
            train = pd.read_csv(train_path)
            
            self.print_status("Loading testing data...", "PROCESSING")
            test = pd.read_csv(test_path)
            
            self.print_status(f"Training data shape: {train.shape}", "INFO")
            self.print_status(f"Testing data shape: {test.shape}", "INFO")
            
            return train, test
        except Exception as e:
            self.print_status(f"Error loading data: {str(e)}", "ERROR")
            sys.exit(1)
            
    def analyze_data(self, train):
        """Perform exploratory data analysis"""
        self.print_status("Performing data analysis...", "PROCESSING")
        
        print(f"\n{Fore.YELLOW}=== DATA OVERVIEW ==={Style.RESET_ALL}")
        print(f"Dataset shape: {train.shape}")
        print(f"Missing values: {train.isnull().sum().sum()}")
        print(f"Duplicate rows: {train.duplicated().sum()}")
        
        print(f"\n{Fore.YELLOW}=== CLASS DISTRIBUTION ==={Style.RESET_ALL}")
        class_dist = train['class'].value_counts()
        for class_name, count in class_dist.items():
            percentage = (count / len(train)) * 100
            print(f"{class_name}: {count} ({percentage:.2f}%)")
            
    def preprocess_data(self, train, test):
        """Preprocess the data"""
        self.print_status("Preprocessing data...", "PROCESSING")
        
        # Label encoding for categorical variables
        def label_encode(df):
            for col in df.columns:
                if df[col].dtype == 'object':
                    label_encoder = LabelEncoder()
                    df[col] = label_encoder.fit_transform(df[col])
        
        label_encode(train)
        label_encode(test)
        
        # Drop unnecessary columns
        if 'num_outbound_cmds' in train.columns:
            train.drop(['num_outbound_cmds'], axis=1, inplace=True)
        if 'num_outbound_cmds' in test.columns:
            test.drop(['num_outbound_cmds'], axis=1, inplace=True)
            
        return train, test
        
    def feature_selection(self, train):
        """Perform feature selection using Random Forest and RFE"""
        self.print_status("Performing feature selection...", "PROCESSING")
        
        X_train = train.drop(['class'], axis=1)
        Y_train = train['class']
        
        rfc = RandomForestClassifier(random_state=42)
        rfe = RFE(rfc, n_features_to_select=10)
        rfe = rfe.fit(X_train, Y_train)
        
        feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
        selected_features = [v for i, v in feature_map if i==True]
        
        self.selected_features = selected_features
        self.print_status(f"Selected features: {', '.join(selected_features)}", "SUCCESS")
        
        return X_train[selected_features], Y_train
        
    def scale_data(self, X_train, test):
        """Scale the features"""
        self.print_status("Scaling features...", "PROCESSING")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        test_scaled = self.scaler.transform(test[self.selected_features])
        
        return X_train_scaled, test_scaled
        
    def split_data(self, X_train, Y_train):
        """Split data into training and validation sets"""
        self.print_status("Splitting data...", "PROCESSING")
        
        x_train, x_test, y_train, y_test = train_test_split(
            X_train, Y_train, train_size=0.70, random_state=42
        )
        
        self.print_status(f"Training set: {x_train.shape}", "INFO")
        self.print_status(f"Validation set: {x_test.shape}", "INFO")
        
        return x_train, x_test, y_train, y_test
        
    def train_models(self, x_train, y_train):
        """Train multiple ML models"""
        self.print_status("Training machine learning models...", "PROCESSING")
        
        # Initialize models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        trained_models = {}
        training_times = {}
        
        for name, model in models.items():
            self.print_status(f"Training {name}...", "PROCESSING")
            start_time = time.time()
            model.fit(x_train, y_train)
            end_time = time.time()
            
            trained_models[name] = model
            training_times[name] = end_time - start_time
            
            self.print_status(f"{name} trained in {training_times[name]:.2f} seconds", "SUCCESS")
            
        return trained_models, training_times
        
    def evaluate_models(self, models, x_train, x_test, y_train, y_test):
        """Evaluate model performance"""
        self.print_status("Evaluating model performance...", "PROCESSING")
        
        results = []
        predictions = {}
        best_score = 0
        
        for name, model in models.items():
            # Training and testing scores
            train_score = model.score(x_train, y_train)
            test_score = model.score(x_test, y_test)
            
            # Predictions
            y_pred = model.predict(x_test)
            predictions[name] = y_pred
            
            # F1 Score
            f1 = f1_score(y_test, y_pred)
            
            results.append([name, f"{train_score:.4f}", f"{test_score:.4f}", f"{f1:.4f}"])
            
            # Track best model
            if test_score > best_score:
                best_score = test_score
                self.best_model = model
                self.best_model_name = name
            
        # Display results table
        headers = ["Model", "Train Score", "Test Score", "F1 Score"]
        print(f"\n{Fore.YELLOW}=== MODEL PERFORMANCE ==={Style.RESET_ALL}")
        print(tabulate(results, headers=headers, tablefmt="fancy_grid"))
        
        return predictions
        
    def save_models(self, models):
        """Save all trained models"""
        self.print_status("Saving trained models...", "PROCESSING")
        
        saved_models = []
        for name, model in models.items():
            metadata = {
                'training_date': datetime.now().isoformat(),
                'features_count': len(self.selected_features),
                'model_type': type(model).__name__
            }
            
            model_path = self.model_manager.save_model(
                model=model,
                model_name=name.replace(' ', '_').lower(),
                scaler=self.scaler,
                features=self.selected_features,
                metadata=metadata
            )
            
            saved_models.append((name, model_path))
            self.print_status(f"Saved {name} to {model_path}", "SUCCESS")
            
        return saved_models
        
    def classify_packet(self, packet_features):
        """Classify a single packet using the best model"""
        if not self.best_model or not self.selected_features:
            return None, 0.0
            
        try:
            # Convert packet features to DataFrame
            feature_df = pd.DataFrame([packet_features])
            
            # Ensure all required features are present
            for feature in self.selected_features:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0
                    
            # Select only the features used in training
            feature_df = feature_df[self.selected_features]
            
            # Scale features
            scaled_features = self.scaler.transform(feature_df)
            
            # Make prediction
            prediction = self.best_model.predict(scaled_features)[0]
            probability = self.best_model.predict_proba(scaled_features)[0].max()
            
            return prediction, probability
            
        except Exception as e:
            print(f"Error classifying packet: {e}")
            return None, 0.0
            
    def start_realtime_monitoring(self, interface=None, duration=60):
        """Start real-time network monitoring"""
        if not SCAPY_AVAILABLE:
            self.print_status("Scapy not available. Cannot start real-time monitoring.", "ERROR")
            return
            
        if not self.best_model:
            self.print_status("No trained model available. Train a model first.", "ERROR")
            return
            
        self.print_status(f"Starting real-time monitoring for {duration} seconds...", "PROCESSING")
        
        # Start packet capture
        self.packet_capture = PacketCapture(interface)
        if not self.packet_capture.start_capture():
            self.print_status("Failed to start packet capture", "ERROR")
            return
            
        start_time = time.time()
        packet_count = 0
        intrusion_count = 0
        
        try:
            while time.time() - start_time < duration:
                packets = self.packet_capture.get_packets()
                
                for packet_data in packets:
                    packet_count += 1
                    features = packet_data['features']
                    timestamp = packet_data['timestamp']
                    
                    # Classify packet
                    prediction, confidence = self.classify_packet(features)
                    
                    if prediction == 1:  # Assuming 1 = intrusion, 0 = normal
                        intrusion_count += 1
                        self.print_status(
                            f"🚨 INTRUSION DETECTED at {timestamp.strftime('%H:%M:%S')} "
                            f"(Confidence: {confidence:.2f})", 
                            "WARNING"
                        )
                        
                time.sleep(1)  # Check every second
                
        except KeyboardInterrupt:
            self.print_status("Monitoring stopped by user", "INFO")
        finally:
            self.packet_capture.stop_capture()
            
        self.print_status(
            f"Monitoring complete. Processed {packet_count} packets, "
            f"detected {intrusion_count} potential intrusions.", 
            "SUCCESS"
        )
        
    def detailed_evaluation(self, models, predictions, y_test):
        """Provide detailed evaluation with confusion matrices"""
        target_names = ["Normal", "Anomaly"]
        
        print(f"\n{Fore.YELLOW}=== DETAILED EVALUATION ==={Style.RESET_ALL}")
        
        for name, y_pred in predictions.items():
            print(f"\n{Fore.CYAN}{'='*20} {name} {'='*20}{Style.RESET_ALL}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n{Fore.GREEN}Confusion Matrix:{Style.RESET_ALL}")
            print(cm)
            
            # Classification Report
            print(f"\n{Fore.GREEN}Classification Report:{Style.RESET_ALL}")
            print(classification_report(y_test, y_pred, target_names=target_names))
            
    def cross_validation(self, models, x_train, y_train):
        """Perform cross-validation"""
        self.print_status("Performing cross-validation...", "PROCESSING")
        
        cv_results = []
        
        for name, model in models.items():
            precision_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='precision')
            recall_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='recall')
            
            precision_mean = precision_scores.mean()
            recall_mean = recall_scores.mean()
            
            cv_results.append([
                name, 
                f"{precision_mean:.4f} ± {precision_scores.std():.4f}",
                f"{recall_mean:.4f} ± {recall_scores.std():.4f}"
            ])
            
        headers = ["Model", "Precision (CV)", "Recall (CV)"]
        print(f"\n{Fore.YELLOW}=== CROSS-VALIDATION RESULTS ==={Style.RESET_ALL}")
        print(tabulate(cv_results, headers=headers, tablefmt="fancy_grid"))
        
    def list_saved_models(self):
        """List all saved models"""
        models = self.model_manager.list_models()
        
        if not models:
            self.print_status("No saved models found", "INFO")
            return
            
        print(f"\n{Fore.YELLOW}=== SAVED MODELS ==={Style.RESET_ALL}")
        headers = ["Model Name", "Timestamp", "Features", "Filename"]
        table_data = []
        
        for model in models:
            table_data.append([
                model['name'],
                model['timestamp'],
                model['features'],
                model['filename']
            ])
            
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
        
    def load_saved_model(self, model_path):
        """Load a previously saved model"""
        self.print_status(f"Loading model from {model_path}...", "PROCESSING")
        
        model_package = self.model_manager.load_model(model_path)
        if model_package:
            self.best_model = model_package['model']
            self.best_model_name = model_package['model_name']
            self.scaler = model_package['scaler']
            self.selected_features = model_package['features']
            
            self.print_status(f"Successfully loaded {self.best_model_name}", "SUCCESS")
            return True
        else:
            self.print_status("Failed to load model", "ERROR")
            return False
        
    def run_analysis(self, train_path, test_path, detailed=False, save_models=True):
        """Run the complete NIDS analysis"""
        self.print_banner()
        
        # Load data
        train, test = self.load_data(train_path, test_path)
        
        # Analyze data
        self.analyze_data(train)
        
        # Preprocess data
        train, test = self.preprocess_data(train, test)
        
        # Feature selection
        X_train, Y_train = self.feature_selection(train)
        
        # Scale data
        X_train_scaled, test_scaled = self.scale_data(X_train, test)
        
        # Split data
        x_train, x_test, y_train, y_test = self.split_data(X_train_scaled, Y_train)
        
        # Store for later use
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        
        # Train models
        models, training_times = self.train_models(x_train, y_train)
        
        # Evaluate models
        predictions = self.evaluate_models(models, x_train, x_test, y_train, y_test)
        
        # Cross-validation
        self.cross_validation(models, x_train, y_train)
        
        # Save models if requested
        if save_models:
            self.save_models(models)
            
        # Detailed evaluation if requested
        if detailed:
            self.detailed_evaluation(models, predictions, y_test)
            
        self.print_status("Analysis completed successfully!", "SUCCESS")
        
        # Display best model
        best_score = self.best_model.score(x_test, y_test)
        print(f"\n{Fore.GREEN}🏆 BEST MODEL: {self.best_model_name} (Accuracy: {best_score:.4f}){Style.RESET_ALL}")
        
        return models

def main():
    parser = argparse.ArgumentParser(
        description="0XVENOM - Enhanced Network Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models
  python 0xvenom.py --train data/train.csv --test data/test.csv
  
  # Train with detailed analysis
  python 0xvenom.py --train data/train.csv --test data/test.csv --detailed
  
  # Start web interface
  python 0xvenom.py --web
  
  # Real-time monitoring
  python 0xvenom.py --monitor --duration 300 --interface eth0
  
  # List saved models
  python 0xvenom.py --list-models
  
  # Load and use saved model for monitoring
  python 0xvenom.py --load-model models/random_forest_20240101_120000.joblib --monitor
        """
    )
    
    parser.add_argument('--train', help='Path to training dataset (CSV file)')
    parser.add_argument('--test', help='Path to testing dataset (CSV file)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed evaluation')
    parser.add_argument('--web', action='store_true', help='Start web interface')
    parser.add_argument('--monitor', action='store_true', help='Start real-time monitoring')
    parser.add_argument('--duration', type=int, default=60, help='Monitoring duration in seconds')
    parser.add_argument('--interface', help='Network interface for packet capture')
    parser.add_argument('--list-models', action='store_true', help='List saved models')
    parser.add_argument('--load-model', help='Load a saved model')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save trained models')
    parser.add_argument('--version', action='version', version='0XVENOM v2.0 - Enhanced NIDS')
    
    args = parser.parse_args()
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Initialize NIDS
    nids = VenomNIDS()
    
    # Handle different modes
    if args.web:
        # Start web interface
        nids.print_status("Starting web interface...", "INFO")
        os.system("streamlit run scripts/web_interface.py")
        
    elif args.list_models:
        # List saved models
        nids.list_saved_models()
        
    elif args.monitor:
        # Real-time monitoring mode
        if args.load_model:
            if nids.load_saved_model(args.load_model):
                nids.start_realtime_monitoring(args.interface, args.duration)
            else:
                sys.exit(1)
        else:
            nids.print_status("No model loaded. Train a model first or use --load-model", "ERROR")
            
    elif args.train and args.test:
        # Training mode
        if not os.path.exists(args.train):
            nids.print_status(f"Training file not found: {args.train}", "ERROR")
            sys.exit(1)
            
        if not os.path.exists(args.test):
            nids.print_status(f"Testing file not found: {args.test}", "ERROR")
            sys.exit(1)
        
        # Run analysis
        models = nids.run_analysis(
            args.train, 
            args.test, 
            detailed=args.detailed,
            save_models=not args.no_save
        )
        
        # Offer real-time monitoring
        if SCAPY_AVAILABLE:
            response = input(f"\n{Fore.CYAN}Start real-time monitoring? (y/n): {Style.RESET_ALL}")
            if response.lower() == 'y':
                nids.start_realtime_monitoring(args.interface, args.duration)
                
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
