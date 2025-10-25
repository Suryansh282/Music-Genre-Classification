import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import torchaudio
import random
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
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
DATA_PATH = '/home/akshay/project_EE708/Music Genre Classification/Train'
SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 5
SEGMENT_DURATION = 6
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

# class MusicGenreDataset(Dataset):
#     """Dataset for music genre classification with on-the-fly feature extraction"""
    
#     def __init__(self, file_paths, labels, segment_indices):
#         """
#         Initialize the dataset.
        
#         Args:
#             file_paths (list): List of audio file paths
#             labels (list): List of labels corresponding to the files
#             segment_indices (list): List of segment indices to extract from each file
#         """
#         self.file_paths = file_paths
#         self.file_paths.remove('/home/akshay/project_EE708/Music Genre Classification/Train/jazz/jazz.00054.wav')
#         self.labels = labels
#         self.segment_indices = segment_indices
        
#     def __len__(self):
#         return len(self.file_paths)
    
#     def __getitem__(self, idx):
#         """
#         Load and process an audio file on-the-fly.
        
#         Args:
#             idx: Index of the item to load
            
#         Returns:
#             tuple: (feature, label) where feature is the processed audio and
#                   label is the class index
#         """
#         file_path = self.file_paths[idx]
#         label = self.labels[idx]
#         segment_idx = self.segment_indices[idx]
        
#         # Extract features from the audio file
#         feature = self.extract_features(file_path, segment_idx)
        
#         # Convert to PyTorch tensors
#         feature_tensor = torch.from_numpy(feature).float()
#         label_tensor = torch.tensor(label, dtype=torch.long)
        
#         # Add channel dimension for CNN
#         if feature_tensor.dim() == 2:
#             feature_tensor = feature_tensor.unsqueeze(0)  # Add channel dimension (C, H, W)
            
#         return feature_tensor, label_tensor
    
#     def extract_features(self, file_path, segment_idx=0):
#         """
#         Extract features from a specific segment of an audio file.
        
#         Args:
#             file_path: Path to the audio file
#             segment_idx: Index of the segment to extract
            
#         Returns:
#             numpy.ndarray: Extracted features
#         """
#         try:
#             # Load audio file
#             y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
#             # Ensure consistent length
#             if len(y) > SAMPLES_PER_TRACK:
#                 y = y[:SAMPLES_PER_TRACK]
#             else:
#                 y = np.pad(y, (0, max(0, SAMPLES_PER_TRACK - len(y))), 'constant')
            
#             # Extract the specific segment
#             start_sample = segment_idx * SEGMENT_SAMPLES
#             end_sample = start_sample + SEGMENT_SAMPLES
#             segment = y[start_sample:end_sample]
            
#             # Extract mel spectrogram
#             mel_spec = librosa.feature.melspectrogram(
#                 y=segment,
#                 sr=SAMPLE_RATE,
#                 n_fft=N_FFT,
#                 hop_length=HOP_LENGTH,
#                 n_mels=128,
#                 fmin=20,
#                 fmax=8000
#             )
            
#             # Convert to log scale (dB)
#             log_mel_spec = librosa.power_to_db(mel_spec)
            
#             # Resize to target dimensions for the CNN
#             log_mel_spec = np.resize(log_mel_spec, IMG_SIZE)
            
#             # Normalize features
#             log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-10)
            
#             return log_mel_spec
            
#         except Exception as e:
#             print(f"Error processing {file_path} (segment {segment_idx}): {str(e)}")
#             # Return a zero array as fallback
#             return np.zeros(IMG_SIZE)

class OnTheFlyMusicGenreDataset(Dataset):
    def __init__(self, file_list, segment_indices, transform=None, mode='train'):
        self.file_list = file_list
        if '/home/akshay/project_EE708/Music Genre Classification/Train/jazz/jazz.00054.wav' in self.file_list:
            self.file_list.remove('/home/akshay/project_EE708/Music Genre Classification/Train/jazz/jazz.00054.wav')
        self.segment_indices = segment_indices
        self.transform = transform
        self.mode = mode
        
        # Audio transforms
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=128,
            f_min=20,
            f_max=8000
        )
        self.amplitude_to_db = AmplitudeToDB()
        
    def __len__(self):
        return len(self.segment_indices)
    
    def _load_audio(self, file_path):
        """Load and preprocess audio file"""
        # print('loading')
        try:
            waveform, sr = torchaudio.load(file_path)
        except:
            print('----------')
            print(file_path)
            print('----------')
            sr = 44100
            return torch.ones((1,int(sr*10)))
        # print('loaded')
        # Resample if necessary
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Ensure correct length
        if waveform.shape[1] < SAMPLES_PER_TRACK:
            padding = SAMPLES_PER_TRACK - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :SAMPLES_PER_TRACK]
            
        return waveform.squeeze()

    def _extract_features(self, waveform):
        """Extract mel spectrogram features"""
        # Generate mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to dB scale
        log_mel_spec = self.amplitude_to_db(mel_spec)
        
        # Resize to target dimensions
        log_mel_spec = torch.tensor(
            np.resize(log_mel_spec.numpy(), IMG_SIZE)
        )
        
        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-10)
        
        return log_mel_spec.unsqueeze(0)  # Add channel dimension

    def __getitem__(self, idx):
        file_idx, segment_idx = self.segment_indices[idx]
        file_path, label = self.file_list[file_idx]
        
        # Load audio
        waveform = self._load_audio(file_path)
        
        # Extract segment
        start_sample = segment_idx * SEGMENT_SAMPLES
        end_sample = start_sample + SEGMENT_SAMPLES
        segment = waveform[start_sample:end_sample]
        
        # Extract features
        features = self._extract_features(segment)
        
        # Apply transforms if available
        if self.transform and self.mode == 'train':
            features = self.transform(features)
            
        return features.float(), torch.tensor(label, dtype=torch.long)



class CNNModel(nn.Module):
    """Hybrid CNN-Transformer with Multi-Head Self Attention"""
    
    def __init__(self, num_classes=len(CLASSES), num_heads=4, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Convolutional Feature Extractor
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2 with attention
            ResidualBlock(64, 128, downsample=True),
            SelfAttentionBlock(128, num_heads),
            
            # Block 3 with attention
            ResidualBlock(128, embed_dim, downsample=True),
            SelfAttentionBlock(embed_dim, num_heads),
            
            nn.Dropout(0.4)
        )
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim*4,
                dropout=0.3,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # Adaptive pooling and classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim*2, num_classes)
        )

    def forward(self, x):
        # Convolutional features
        x = self.conv_layers(x)  # [B, 256, H', W']
        
        # Prepare for transformer
        b, c, h, w = x.size()
        x = x.view(b, c, h*w).permute(0, 2, 1)  # [B, seq_len, embed_dim]
        
        # Transformer processing
        x = self.transformer(x)
        
        # Pooling and classification
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)

class ResidualBlock(nn.Module):
    """Residual block with depthwise separable convolutions"""
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class SelfAttentionBlock(nn.Module):
    """Multi-head self attention with positional encoding"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads,
            dropout=0.2,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim*4, embed_dim)
        )
        
        # Learnable scale parameter for attention contribution
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, C, H, W = x.size()
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)
        
        # Self-attention with residual
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x = x + self.alpha * attn_out.permute(0, 2, 1).view_as(x)
        
        # Feedforward with residual
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)
        x = x + self.ffn(self.norm(x_flat)).permute(0, 2, 1).view_as(x)
        return x


def prepare_datasets(data_path):
    # Collect all audio files with their labels
    file_list = []
    for class_name in CLASSES:
        class_dir = os.path.join(data_path, class_name)
        if os.path.isdir(class_dir):
            for f in os.listdir(class_dir):
                if f.endswith('.wav'):
                    file_path = os.path.join(class_dir, f)
                    file_list.append((file_path, CLASS_MAPPING[class_name]))
    
    # Create segment indices for stratified split
    np.random.shuffle(file_list)
    file_indices = list(range(len(file_list)))
    
    # Split files into train/val/test (70/15/15)
    train_files = []
    val_files = []
    test_files = []
    
    for class_idx in range(len(CLASSES)):
        class_files = [f for f in file_list if f[1] == class_idx]
        n = len(class_files)
        n_train = int(n * 0.95)
        n_val = int(n * 0.02)
        
        train_files.extend(class_files[:n_train])
        val_files.extend(class_files[n_train:n_train+n_val])
        test_files.extend(class_files[n_train+n_val:])
    
    # Generate segment indices for each split
    def generate_segment_indices(files):
        indices = []
        for file_idx, (file_path, label) in enumerate(files):
            for seg_idx in range(NUM_SEGMENTS):
                indices.append((file_idx, seg_idx))
        return indices
    
    train_segment_indices = generate_segment_indices(train_files)
    val_segment_indices = generate_segment_indices(val_files)
    test_segment_indices = generate_segment_indices(test_files)
    
    # Create datasets with mixup augmentation
    train_dataset = OnTheFlyMusicGenreDataset(
        train_files,
        train_segment_indices,
        transform=None,  # Add other transforms if needed
        mode='train'
    )
    
    val_dataset = OnTheFlyMusicGenreDataset(
        val_files,
        val_segment_indices,
        mode='val'
    )
    
    test_dataset = OnTheFlyMusicGenreDataset(
        test_files,
        test_segment_indices,
        mode='test'
    )
    
    return train_dataset, val_dataset, test_dataset

def custom_collate(batch):
    """Custom collate function to apply mixup"""
    mixup = Mixup(alpha=0.4)
    return mixup(batch)


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

# def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
#     """Train the model and return training history"""
#     best_val_acc = 0.0
#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []
    
#     for epoch in range(epochs):
#         # Training phase
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             # Zero the parameter gradients
#             optimizer.zero_grad()
            
#             # Forward + backward + optimize
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             # Statistics
#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = correct / total
#         train_losses.append(epoch_loss)
#         train_accs.append(epoch_acc)
        
#         # Validation phase
#         model.eval()
#         val_running_loss = 0.0
#         val_correct = 0
#         val_total = 0
        
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
                
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
                
#                 # Statistics
#                 val_running_loss += loss.item() * inputs.size(0)
#                 _, predicted = torch.max(outputs, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()
        
#         val_epoch_loss = val_running_loss / len(val_loader.dataset)
#         val_epoch_acc = val_correct / val_total
#         val_losses.append(val_epoch_loss)
#         val_accs.append(val_epoch_acc)
        
#         # Update learning rate
#         scheduler.step(val_epoch_loss)
        
#         # Print statistics
#         print(f'Epoch {epoch+1}/{epochs}, '
#               f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
#               f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
        
#         # Save best model
#         if val_epoch_acc > best_val_acc:
#             best_val_acc = val_epoch_acc
#             torch.save(model.state_dict(), 'best_model.pth')
#             print(f"Saved new best model with validation accuracy: {best_val_acc:.4f}")
    
#     # Load best model
#     model.load_state_dict(torch.load('best_model.pth'))
    
#     return model, train_losses, val_losses, train_accs, val_accs

# def evaluate_model(model, test_loader, device):
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

# class Mixup:
    """Mixup augmentation implementation"""
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        
    def __call__(self, batch):
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        
        # Generate mixup coefficients
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size)
        
        # Mix inputs
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        
        # Mix labels (convert to one-hot)
        labels_onehot = F.one_hot(labels, num_classes=len(CLASSES)).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]
        
        return mixed_inputs, mixed_labels

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Handle mixup labels
            is_mixup = labels.dim() > 1
            if is_mixup:
                # For mixup, use KLDivLoss with log probabilities
                outputs = model(inputs)
                log_probs = F.log_softmax(outputs, dim=1)
                loss = F.kl_div(log_probs, labels, reduction='batchmean')
                
                # For accuracy calculation, use dominant class
                class_labels = labels.argmax(dim=1)
            else:
                # Normal training
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                class_labels = labels
                
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            _, predicted = torch.max(outputs.data, 1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += (predicted == class_labels).sum().item()
            total_samples += inputs.size(0)
            
            # Progress update
            if (batch_idx + 1) % 50 == 0:
                current_loss = running_loss / total_samples
                current_acc = running_correct / total_samples
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {current_loss:.4f}, Acc: {current_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_correct += (predicted == labels).sum().item()
                val_total += inputs.size(0)
        
        # Calculate epoch metrics
        train_loss = running_loss / total_samples
        train_acc = running_correct / total_samples
        val_loss = val_running_loss / val_total
        val_acc = val_running_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with val_loss: {val_loss:.4f}')
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc']

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    report = classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, cm

class Mixup:
    """Improved Mixup implementation with proper label handling"""
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        
    def __call__(self, batch):
        inputs, labels = zip(*batch)
        inputs = torch.stack(inputs)
        labels = torch.stack(labels)
        
        # Generate mixup coefficients
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = inputs.size(0)
        index = torch.randperm(batch_size)
        
        # Mix inputs
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        
        # Convert labels to one-hot
        labels_onehot = F.one_hot(labels, num_classes=len(CLASSES)).float()
        
        # Mix labels
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]
        
        return mixed_inputs, mixed_labels

# def main():
#     print("Starting music genre classification with PyTorch...")
    
#     # Collect file paths and labels
#     file_paths = []
#     labels = []
#     segment_indices = []
#     metadata = []
    
#     print("Collecting audio file paths...")
#     for class_name in CLASSES:
#         class_dir = os.path.join(DATA_PATH, class_name)
        
#         if not os.path.isdir(class_dir):
#             print(f"Warning: Class directory '{class_dir}' not found")
#             continue
            
#         print(f"Processing class: {class_name}")
        
#         # Get all WAV files in the class directory
#         wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        
#         if len(wav_files) == 0:
#             print(f"Warning: No WAV files found in {class_dir}")
#             continue
            
#         print(f"Found {len(wav_files)} files in {class_name}")
        
#         # Collect file paths and labels
#         for f in wav_files:
#             file_path = os.path.join(class_dir, f)
            
#             # For each audio file, we will create NUM_SEGMENTS data points
#             for segment_idx in range(NUM_SEGMENTS):
#                 file_paths.append(file_path)
#                 labels.append(CLASS_MAPPING[class_name])
#                 segment_indices.append(segment_idx)
#                 metadata.append({
#                     'file': f,
#                     'class': class_name,
#                     'segment': segment_idx
#                 })
    
#     print(f"Total samples: {len(file_paths)} ({len(file_paths) // NUM_SEGMENTS} files × {NUM_SEGMENTS} segments)")
    
#     # Create custom sampler to ensure all segments from the same file go to the same split
#     # First, group indices by file
#     file_groups = {}
#     for i, meta in enumerate(metadata):
#         file_key = f"{meta['class']}_{meta['file']}"
#         if file_key not in file_groups:
#             file_groups[file_key] = []
#         file_groups[file_key].append(i)
    
#     # Group file indices by class
#     files_by_class = {}
#     for file_key, indices in file_groups.items():
#         class_name = file_key.split('_')[0]
#         if class_name not in files_by_class:
#             files_by_class[class_name] = []
#         files_by_class[class_name].append(indices)
    
#     # Split files into train, validation, and test sets while maintaining class balance
#     train_indices = []
#     val_indices = []
#     test_indices = []
    
#     for class_name, file_indices_list in files_by_class.items():
#         # Shuffle the files for this class
#         np.random.shuffle(file_indices_list)
        
#         # Split: 70% train, 15% validation, 15% test
#         n_files = len(file_indices_list)
#         n_train = int(n_files * 0.7)
#         n_val = int(n_files * 0.15)
        
#         # Add all segments from each file to the appropriate set
#         for i, file_indices in enumerate(file_indices_list):
#             if i < n_train:
#                 train_indices.extend(file_indices)
#             elif i < n_train + n_val:
#                 val_indices.extend(file_indices)
#             else:
#                 test_indices.extend(file_indices)
    
#     # Create datasets for each split
#     train_dataset = MusicGenreDataset(
#         [file_paths[i] for i in train_indices],
#         [labels[i] for i in train_indices],
#         [segment_indices[i] for i in train_indices]
#     )
    
#     val_dataset = MusicGenreDataset(
#         [file_paths[i] for i in val_indices],
#         [labels[i] for i in val_indices],
#         [segment_indices[i] for i in val_indices]
#     )
    
#     test_dataset = MusicGenreDataset(
#         [file_paths[i] for i in test_indices],
#         [labels[i] for i in test_indices],
#         [segment_indices[i] for i in test_indices]
#     )
    
#     print(f"Training set: {len(train_dataset)} samples")
#     print(f"Validation set: {len(val_dataset)} samples")
#     print(f"Test set: {len(test_dataset)} samples")
    
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=BATCH_SIZE, 
#         shuffle=True, 
#         num_workers=4, 
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=BATCH_SIZE, 
#         shuffle=False, 
#         num_workers=4, 
#         pin_memory=True
#     )
    
#     test_loader = DataLoader(
#         test_dataset, 
#         batch_size=BATCH_SIZE, 
#         shuffle=False, 
#         num_workers=4, 
#         pin_memory=True
#     )
    
#     # Build the model
#     # print("Building CNN model...")
#     model = CNNModel(num_classes=len(CLASSES))
#     model = model.to(device)
#     # print(model)
    
#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
#     # Learning rate scheduler
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, 
#         mode='min', 
#         factor=0.5, 
#         patience=5, 
#         min_lr=1e-6, 
#         verbose=True
#     )
    
#     # Train the model
#     print("Training the model...")
#     model, train_losses, val_losses, train_accs, val_accs = train_model(
#         model, 
#         train_loader, 
#         val_loader, 
#         criterion, 
#         optimizer, 
#         scheduler, 
#         EPOCHS, 
#         device
#     )
    
#     # Save the training history
#     save_train_history(train_losses, val_losses, train_accs, val_accs)
    
#     # Save the final model
#     torch.save(model.state_dict(), "cnn_model.pth")
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'class_names': CLASSES,
#         'img_size': IMG_SIZE,
#     }, "cnn_model_full.pth")
    
#     # Evaluate on test set
#     print("Evaluating on test set...")
#     test_acc, report, cm = evaluate_model(model, test_loader, device)
#     print(f"Test accuracy: {test_acc:.4f}")
    
#     # Save model architecture as text file
#     with open('model_summary.txt', 'w') as f:
#         f.write(str(model))
    
#     print("Model training and evaluation complete!")

# if __name__ == "__main__":
#     main()

def main():
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(DATA_PATH)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model setup (same as original)
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6, 
        verbose=True
    )
    
    # Training loop (same as original)
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, device
    )
    
    # Evaluation (same as original)
    test_acc, report, cm = evaluate_model(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
