"""
Evaluation script for Music Genre Classification model (PyTorch version)
Usage: python evaluate.py --test_path /path/to/test/data
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Set memory usage behavior
    torch.cuda.empty_cache()
else:
    print("No GPUs found. Running on CPU.")

# Configuration
SAMPLE_RATE = 22050
DURATION = 30  # seconds (full track duration)
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 6  # Split each 30-second track into 6 segments of 5 seconds each
SEGMENT_DURATION = 5  # Each segment is 5 seconds
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

# Class names
CLASSES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

class CNNModel(nn.Module):
    """CNN model for music genre classification"""
    
    def __init__(self, num_classes=len(CLASSES)):
        super(CNNModel, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4)
        )
        
        # Calculate the flattened size - depends on input size and pooling
        # For 256x256 input with 4 max pooling layers (factor of 16 reduction)
        flattened_size = 256 * (IMG_SIZE[0] // 16) * (IMG_SIZE[1] // 16)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Apply convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Apply fully connected layers
        x = self.fc(x)
        return x

class TestDataset(Dataset):
    """Dataset for test audio files"""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # Add channel dimension if not present (for CNN)
        if feature.dim() == 2:
            feature = feature.unsqueeze(0)  # Add channel dimension (C, H, W)
        return feature, label

def extract_features(file_path):
    """
    Extract features from audio file and convert to spectrogram image.
    For each 30-second audio file, create 6 segments of 5 seconds each.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Ensure consistent length (30 seconds)
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            y = np.pad(y, (0, max(0, SAMPLES_PER_TRACK - len(y))), 'constant')
        
        features = []
        
        # Process each 5-second segment from the 30-second file
        for s in range(NUM_SEGMENTS):
            start_sample = s * SEGMENT_SAMPLES
            end_sample = start_sample + SEGMENT_SAMPLES
            
            segment = y[start_sample:end_sample]
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=segment,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=128,
                fmin=20,
                fmax=8000
            )
            
            # Convert to log scale (dB)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # Resize to target dimensions for the CNN
            log_mel_spec = np.resize(log_mel_spec, IMG_SIZE)
            
            # Normalize features
            log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-10)
            
            features.append(log_mel_spec)
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []

def process_test_data(test_path):
    """
    Process all audio files in test directory
    """
    features = []
    labels = []
    file_paths = []
    segment_info = []  # Track which segment each feature comes from
    
    # Loop through each class folder in test directory
    for class_name in CLASSES:
        class_dir = os.path.join(test_path, class_name)
        
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory {class_dir} not found")
            continue
            
        print(f"Processing class: {class_name}")
        
        # Get all WAV files in the class directory
        wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        print(f"Found {len(wav_files)} files in {class_name}")
        
        # Process each WAV file in the class directory
        for filename in wav_files:
            file_path = os.path.join(class_dir, filename)
            
            # Extract features for segments
            data = extract_features(file_path)
            
            # Add features and labels
            for segment_idx, feature in enumerate(data):
                features.append(feature)
                labels.append(CLASSES.index(class_name))
                file_paths.append(file_path)
                segment_info.append({
                    'file': filename,
                    'class': class_name,
                    'segment': segment_idx
                })
    
    print(f"Total features extracted from test data: {len(features)}")
    
    return np.array(features), np.array(labels), file_paths, segment_info

def main():
    parser = argparse.ArgumentParser(description='Evaluate music genre classification model')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--model_path', type=str, default='cnn_model.pth', help='Path to the trained model')
    args = parser.parse_args()
    
    # Process test data
    print(f"Processing test data from {args.test_path}...")
    features, labels, file_paths, segment_info = process_test_data(args.test_path)
    
    # Create dataset and dataloader
    test_dataset = TestDataset(features, labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = CNNModel(num_classes=len(CLASSES))
    
    # Try to load model in different formats (just state_dict or full checkpoint)
    try:
        # First try loading as full checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dictionary")
    except Exception as e:
        # Try loading just the state_dict
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print("Loaded model state dictionary")
        except Exception as inner_e:
            print(f"Error loading model: {inner_e}")
            return
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Evaluate the model
    print("Evaluating model...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Collect predictions and true labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    pred_labels = np.array(all_predictions)
    true_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")
    f1 = f1_score(true_labels, pred_labels, average="weighted")
    
    # Print evaluation metrics
    print(f"\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Generate classification report
    report = classification_report(true_labels, pred_labels, target_names=CLASSES)
    print("\nClassification Report:\n", report)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("test_confusion_matrix.png")
    
    # Calculate class-wise accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"Accuracy for class {CLASSES[i]}: {acc:.4f}")
    
    # Find misclassified examples
    misclassified_indices = np.where(pred_labels != true_labels)[0]
    if len(misclassified_indices) > 0:
        print(f"\nSample misclassifications (showing up to 5):")
        for i in misclassified_indices[:min(5, len(misclassified_indices))]:
            true_class = CLASSES[true_labels[i]]
            pred_class = CLASSES[pred_labels[i]]
            file_path = file_paths[i]
            print(f"File: {os.path.basename(file_path)}, True: {true_class}, Predicted: {pred_class}")

if __name__ == "__main__":
    main()