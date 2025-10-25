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

# Check GPU availability
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

# Configuration
# DATA_PATH = 'Train'  # Update this with your dataset path
DATA_PATH = '/home/akshay/project_EE708/Music Genre Classification/Train'  # Update this with your dataset path
SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_SEGMENTS = 10  # Increased from 6 to 10 for more data
SEGMENT_DURATION = 3  # Reduced to 3 seconds for more diversity
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION
BATCH_SIZE = 32
EPOCHS = 100  # Increased for better convergence
IMG_SIZE = (256, 256)
LEARNING_RATE = 3e-4  # Adjusted learning rate
MIXUP_ALPHA = 0.2  # For mixup augmentation

# Mapping class names to indices
CLASSES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]
CLASS_MAPPING = {class_name: i for i, class_name in enumerate(CLASSES)}

# Feature extraction parameters
N_MFCC = 40  # Increased from 13
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000


class MusicGenreDataset(Dataset):
    """Dataset for music genre classification with advanced transforms"""
    
    def __init__(self, features, labels, transform=None, is_training=False):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Add channel dimension if not present (for CNN)
        if feature.dim() == 2:
            feature = feature.unsqueeze(0)
            
        # Apply transforms if specified (for data augmentation)
        if self.transform and self.is_training:
            feature = self.transform(feature)
            
        return feature, label


def extract_multi_features(file_path):
    """
    Extract multiple feature types from audio
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Ensure consistent length
        if len(y) > SAMPLES_PER_TRACK:
            y = y[:SAMPLES_PER_TRACK]
        else:
            y = np.pad(y, (0, max(0, SAMPLES_PER_TRACK - len(y))), 'constant')
        
        multi_features = []
        
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
                n_mels=N_MELS,
                fmin=FMIN,
                fmax=FMAX
            )
            
            # Convert to log scale (dB)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=segment, 
                sr=SAMPLE_RATE, 
                n_mfcc=N_MFCC,
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH
            )
            
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(
                y=segment, 
                sr=SAMPLE_RATE,
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH
            )
            
            # Extract spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=segment,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
            
            # Resize to target dimensions for the CNN
            log_mel_spec = np.resize(log_mel_spec, IMG_SIZE)
            
            # Normalize features
            log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-10)
            
            # Create a 3-channel image by stacking different time-frequency representations
            channel1 = log_mel_spec
            
            # Create a tempo-normalized spectrogram for channel 2
            tempo, _ = librosa.beat.beat_track(y=segment, sr=SAMPLE_RATE)
            if tempo > 0:
                tempo_factor = 120.0 / tempo  # Normalize to 120 BPM
            else:
                tempo_factor = 1.0
                
            segment_tempo_adjusted = librosa.effects.time_stretch(segment, rate=tempo_factor)
            if len(segment_tempo_adjusted) > SEGMENT_SAMPLES:
                segment_tempo_adjusted = segment_tempo_adjusted[:SEGMENT_SAMPLES]
            else:
                segment_tempo_adjusted = np.pad(segment_tempo_adjusted, 
                                              (0, max(0, SEGMENT_SAMPLES - len(segment_tempo_adjusted))), 
                                              'constant')
                
            mel_spec_tempo = librosa.feature.melspectrogram(
                y=segment_tempo_adjusted,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )
            channel2 = librosa.power_to_db(mel_spec_tempo)
            channel2 = np.resize(channel2, IMG_SIZE)
            channel2 = (channel2 - np.mean(channel2)) / (np.std(channel2) + 1e-10)
            
            # For channel 3, use harmonic-percussive source separation
            harmonic, percussive = librosa.effects.hpss(segment)
            mel_spec_percussive = librosa.feature.melspectrogram(
                y=percussive,
                sr=SAMPLE_RATE,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )
            channel3 = librosa.power_to_db(mel_spec_percussive)
            channel3 = np.resize(channel3, IMG_SIZE)
            channel3 = (channel3 - np.mean(channel3)) / (np.std(channel3) + 1e-10)
            
            # Stack the three channels
            feature_image = np.stack([channel1, channel2, channel3], axis=0)
            
            multi_features.append(feature_image)
        
        return multi_features
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []


def process_audio_files(data_path):
    """Process all audio files with multi-feature extraction"""
    features = []
    labels = []
    file_info = []
    
    for class_name in CLASSES:
        class_dir = os.path.join(data_path, class_name)
        
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory '{class_dir}' not found")
            continue
            
        print(f"Processing class: {class_name}")
        
        wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        
        if len(wav_files) == 0:
            print(f"Warning: No WAV files found in {class_dir}")
            continue
            
        print(f"Found {len(wav_files)} files in {class_name}")
        
        for file_idx, f in enumerate(wav_files):
            file_path = os.path.join(class_dir, f)
            
            # Extract multiple features for all segments
            data = extract_multi_features(file_path)
            
            for segment_idx, feature in enumerate(data):
                features.append(feature)
                labels.append(CLASS_MAPPING[class_name])
                file_info.append({
                    'file': f,
                    'class': class_name,
                    'segment': segment_idx
                })
    
    print(f"Total features extracted: {len(features)}")
    
    return np.array(features), np.array(labels), file_info


class AttentionBlock(nn.Module):
    """Self-attention block for focusing on important features"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Reshape for attention calculation
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        out = self.gamma * out + x
        return out


class EnhancedCNNModel(nn.Module):
    """Enhanced CNN with residual connections and attention"""
    
    def __init__(self, num_classes=len(CLASSES)):
        super(EnhancedCNNModel, self).__init__()
        
        # First convolutional block with residual connection
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Now using 3-channel input
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # Residual block 1
        self.res1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Attention block
        self.attention = AttentionBlock(128)
        
        # Residual block 2
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Residual block 3
        self.res3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256)
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Global average pooling 
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Apply convolutional blocks with residual connections
        x = self.conv1(x)
        
        # Residual connection 1
        identity = x
        x = self.res1(x) + identity
        
        x = self.conv2(x)
        x = self.attention(x)  # Apply attention
        
        # Residual connection 2
        identity = x
        x = self.res2(x) + identity
        
        x = self.conv3(x)
        
        # Residual connection 3
        identity = x
        x = self.res3(x) + identity
        
        x = self.conv4(x)
        
        x = self.global_pool(x)
        x = self.fc(x)
        return x


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Criterion for mixup training"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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


def train_with_mixup(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    """Train with mixup data augmentation for robustness"""
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
            
            # Mixup data
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics - for mixup, use the primary labels for calculating accuracy
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        
        # Validation phase - no mixup for validation
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
                val_correct += predicted.eq(labels).sum().item()
        
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


def create_data_splits(features, labels, file_info):
    """Create stratified data splits"""
    # Group by file to keep segments from the same file together
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    print("Starting enhanced music genre classification with PyTorch...")
    
    # Extract multi-features from dataset
    print("Extracting multi-features from audio files...")
    features, labels, file_info = process_audio_files(DATA_PATH)
    
    # Create dataset splits
    X_train, y_train, X_val, y_val, X_test, y_test = create_data_splits(features, labels, file_info)
    
    # Define data augmentation transforms
    train_transform = lambda x: torchaudio.transforms.FrequencyMasking(freq_mask_param=30)(x) if random.random() > 0.5 else x
    
    # Create datasets and dataloaders
    train_dataset = MusicGenreDataset(X_train, y_train, transform=train_transform, is_training=True)
    val_dataset = MusicGenreDataset(X_val, y_val)
    test_dataset = MusicGenreDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Build the enhanced CNN model
    print("Building enhanced CNN model...")
    model = EnhancedCNNModel(num_classes=len(CLASSES))
    model = model.to(device)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Increase restart period by factor of 2 at each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Train the model with mixup augmentation
    print("Training the enhanced model...")
    model, train_losses, val_losses, train_accs, val_accs = train_with_mixup(
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
    torch.save(model.state_dict(), "enhanced_cnn_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': CLASSES,
        'img_size': IMG_SIZE,
    }, "enhanced_cnn_model_full.pth")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_acc, report, cm = evaluate_model(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model architecture as text file
    with open('model_summary.txt', 'w') as f:
        f.write(str(model))
    
    # Analysis of most difficult classes
    class_accuracies = []
    class_predictions = {}
    
    for i, class_name in enumerate(CLASSES):
        class_indices = np.where(np.array(all_labels) == i)[0]
        class_preds = np.array(all_predictions)[class_indices]
        class_true = np.array(all_labels)[class_indices]
        class_acc = np.mean(class_preds == class_true)
        class_accuracies.append((class_name, class_acc))
        
        # Store predictions for this class
        class_predictions[class_name] = {
            'predictions': class_preds,
            'true_labels': class_true
        }
    
    # Sort by accuracy
    class_accuracies.sort(key=lambda x: x[1])
    
    # Print most difficult classes
    print("\nClass accuracies (sorted from most difficult to easiest):")
    for class_name, acc in class_accuracies:
        print(f"{class_name}: {acc:.4f}")
    
    # Create bar chart for class accuracies
    plt.figure(figsize=(12, 6))
    classes, accs = zip(*class_accuracies)
    bars = plt.bar(classes, accs, color='skyblue')
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}', ha='center', va='bottom')
    
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("class_accuracies.png")
    
    print("Model training and evaluation complete!")

if __name__ == "__main__":
    main()