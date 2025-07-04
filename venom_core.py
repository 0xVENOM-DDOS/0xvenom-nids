#!/usr/bin/env python3
"""
Core classes for 0XVENOM NIDS
Separated for better modularity and web interface integration
"""

import os
import sys
import time
import json
import joblib
import threading
import queue
from datetime import datetime
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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

# Initialize colorama
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
        
        if SCAPY_AVAILABLE and packet.haslayer(IP):
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
        if not os.path.exists(self.models_dir):
            return models
            
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
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            return train, test
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
            
    def preprocess_data(self, train, test):
        """Preprocess the data"""
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
        X_train = train.drop(['class'], axis=1)
        Y_train = train['class']
        
        rfc = RandomForestClassifier(random_state=42)
        rfe = RFE(rfc, n_features_to_select=10)
        rfe = rfe.fit(X_train, Y_train)
        
        feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns)]
        selected_features = [v for i, v in feature_map if i==True]
        
        self.selected_features = selected_features
        return X_train[selected_features], Y_train
        
    def scale_data(self, X_train, test):
        """Scale the features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        test_scaled = self.scaler.transform(test[self.selected_features])
        return X_train_scaled, test_scaled
        
    def split_data(self, X_train, Y_train):
        """Split data into training and validation sets"""
        x_train, x_test, y_train, y_test = train_test_split(
            X_train, Y_train, train_size=0.70, random_state=42
        )
        return x_train, x_test, y_train, y_test
        
    def train_models(self, x_train, y_train):
        """Train multiple ML models"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        trained_models = {}
        
        for name, model in models.items():
            model.fit(x_train, y_train)
            trained_models[name] = model
            
        return trained_models, {}
        
    def evaluate_models(self, models, x_train, x_test, y_train, y_test):
        """Evaluate model performance"""
        predictions = {}
        best_score = 0
        
        for name, model in models.items():
            test_score = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            predictions[name] = y_pred
            
            if test_score > best_score:
                best_score = test_score
                self.best_model = model
                self.best_model_name = name
            
        return predictions
        
    def save_models(self, models):
        """Save all trained models"""
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
            
        return saved_models
        
    def load_saved_model(self, model_path):
        """Load a previously saved model"""
        model_package = self.model_manager.load_model(model_path)
        if model_package:
            self.best_model = model_package['model']
            self.best_model_name = model_package['model_name']
            self.scaler = model_package['scaler']
            self.selected_features = model_package['features']
            return True
        return False
        
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
            return None, 0.0
        
    def run_analysis(self, train_path, test_path, detailed=False, save_models=True):
        """Run the complete NIDS analysis"""
        # Load data
        train, test = self.load_data(train_path, test_path)
        
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
        
        # Save models if requested
        if save_models:
            self.save_models(models)
            
        return models
