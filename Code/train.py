import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import librosa
import random
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Set memory usage behavior (equivalent to TF memory growth setting)
    torch.cuda.empty_cache()
    # Print available memory
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No GPUs found. Running on CPU.")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
DATA_PATH = '/home/akshay/project_EE708/Music Genre Classification/Train'  # Update this with your dataset path
SAMPLE_RATE = 22050
DURATION = 30  # seconds (full length of tracks according to dataset stats)
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 5  # Split each 30-second track into 6 segments of 5 seconds each
SEGMENT_DURATION = 6  # Each segment is 5 seconds
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION
BATCH_SIZE = 64
EPOCHS = 50  
IMG_SIZE = (256, 256)
LEARNING_RATE = 0.0007

# Mapping class names to indices
CLASSES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]
CLASS_MAPPING = {class_name: i for i, class_name in enumerate(CLASSES)}

class MusicGenreDataset(Dataset):
    """Dataset for music genre classification"""
    
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

def process_all_audio_files(data_path):
    """
    Process all audio files and extract features.
    For each class directory, process all files and create segments per file.
    """
    features = []
    labels = []
    file_info = []  # To track which file and segment each feature comes from
    
    # Process each class directory
    for class_name in CLASSES:
        class_dir = os.path.join(data_path, class_name)
        
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory '{class_dir}' not found")
            continue
            
        print(f"Processing class: {class_name}")
        
        # Get all WAV files in the class directory
        wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        
        if len(wav_files) == 0:
            print(f"Warning: No WAV files found in {class_dir}")
            continue
            
        print(f"Found {len(wav_files)} files in {class_name}")
        
        # Process each WAV file
        for file_idx, f in enumerate(wav_files):
            file_path = os.path.join(class_dir, f)
            
            # Extract features for all segments in this file
            data = extract_features(file_path)
            
            # Add features and labels
            for segment_idx, feature in enumerate(data):
                features.append(feature)
                labels.append(CLASS_MAPPING[class_name])
                file_info.append({
                    'file': f,
                    'class': class_name,
                    'segment': segment_idx
                })
    
    print(f"Total features extracted: {len(features)}")
    print(f"Expected: {len(CLASSES) * 80 * NUM_SEGMENTS} (10 classes × 80 files × {NUM_SEGMENTS} segments)")
    
    return np.array(features), np.array(labels), file_info

def extract_features(file_path):
    """
    Extract features from audio file and convert to spectrogram image.
    For each audio file, create segments.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Ensure consistent length
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            y = np.pad(y, (0, max(0, SAMPLES_PER_TRACK - len(y))), 'constant')
        
        features = []
        
        # Process each segment from the file
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
        # Resulting feature map size: 256/16 = 16
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

def save_train_history(train_losses, val_losses, train_accs, val_accs, filename='training_history.png'):
    """Save training history as a plot"""
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    """Train the model and return training history"""
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        # Update learning rate
        scheduler.step(val_epoch_loss)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
        
        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with validation accuracy: {best_val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    # Generate classification report
    report = classification_report(
        all_labels, 
        all_predictions, 
        target_names=CLASSES, 
        output_dict=True
    )
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=CLASSES))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    
    return accuracy, report, cm

def main():
    print("Starting music genre classification with PyTorch...")
    
    # Extract features from dataset
    print("Extracting features from audio files...")
    features, labels, file_info = process_all_audio_files(DATA_PATH)
    
    # Create indices for stratified split that keeps segments from the same file together
    unique_files = {}
    for i, info in enumerate(file_info):
        file_key = f"{info['class']}_{info['file']}"
        if file_key not in unique_files:
            unique_files[file_key] = []
        unique_files[file_key].append(i)
    
    # Get file indices by class
    files_by_class = {}
    for file_key, indices in unique_files.items():
        class_name = file_key.split('_')[0]
        if class_name not in files_by_class:
            files_by_class[class_name] = []
        files_by_class[class_name].append(indices)
    
    # Split files into train, validation, and test sets while maintaining class balance
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_name, file_indices_list in files_by_class.items():
        # Shuffle the files for this class
        np.random.shuffle(file_indices_list)
        
        # Split: 70% train, 15% validation, 15% test
        n_files = len(file_indices_list)
        n_train = int(n_files * 0.7)
        n_val = int(n_files * 0.15)
        
        # Add all segments from each file to the appropriate set
        for i, file_indices in enumerate(file_indices_list):
            if i < n_train:
                train_indices.extend(file_indices)
            elif i < n_train + n_val:
                val_indices.extend(file_indices)
            else:
                test_indices.extend(file_indices)
    
    # Create dataset splits
    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_val = features[val_indices]
    y_val = labels[val_indices]
    X_test = features[test_indices]
    y_test = labels[test_indices]
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Create datasets and dataloaders
    train_dataset = MusicGenreDataset(X_train, y_train)
    val_dataset = MusicGenreDataset(X_val, y_val)
    test_dataset = MusicGenreDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Build the model
    print("Building CNN model...")
    model = CNNModel(num_classes=len(CLASSES))
    model = model.to(device)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6, 
        verbose=True
    )
    
    # Train the model
    print("Training the model...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        EPOCHS, 
        device
    )
    
    # Save the training history
    save_train_history(train_losses, val_losses, train_accs, val_accs)
    
    # Save the final model
    torch.save(model.state_dict(), "cnn_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': CLASSES,
        'img_size': IMG_SIZE,
    }, "cnn_model_full.pth")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_acc, report, cm = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model architecture as text file
    with open('model_summary.txt', 'w') as f:
        f.write(str(model))
    
    print("Model training and evaluation complete!")

if __name__ == "__main__":
    main()