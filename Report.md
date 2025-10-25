# Music Genre Classification using Convolutional Neural Networks on Mel-Spectrograms

## Abstract
This paper presents a deep learning approach for automatic music genre classification using convolutional neural networks (CNNs). We developed a system that processes audio files, extracts mel-spectrogram features, and classifies them into one of ten music genres. Our model achieved an overall accuracy of 71.35% on the test set, with particularly strong performance on classical and metal genres. We discuss the preprocessing techniques, model architecture, and evaluation results, along with potential improvements for future work.

## I. Introduction

Music genre classification is an important task in music information retrieval (MIR) that involves categorizing music tracks into predefined genres based on their audio content. Automatic genre classification can facilitate music organization, recommendation systems, and music discovery platforms. In this study, we implement a deep learning approach using convolutional neural networks to classify audio files into ten different music genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

The objective of this research is to:
1. Develop a robust preprocessing pipeline for audio features extraction
2. Design and train an effective CNN architecture for genre classification
3. Evaluate the model's performance across different genres
4. Identify challenges and propose potential improvements

## II. Methodology

### A. Dataset

We utilized a dataset comprising audio files categorized into ten distinct music genres. Each genre contains approximately 80 audio samples, with each sample being a 30-second track. The audio files were originally sampled at 22,050 Hz.

### B. Preprocessing Pipeline

Our preprocessing approach involved several key steps:

1) **Audio Segmentation**: Each 30-second audio file was divided into 6 segments of 5 seconds each to increase the training sample size and capture temporal variations within tracks.

2) **Feature Extraction**: For each segment, we extracted mel-spectrograms using the following parameters:
   - Sample rate: 22,050 Hz
   - FFT window size: 2,048 samples
   - Hop length: 512 samples
   - Number of mel bands: 128
   - Frequency range: 20-8,000 Hz

3) **Spectrogram Processing**: The mel-spectrograms were converted to logarithmic scale (dB) and resized to 256×256 dimensions to maintain a consistent input size for the CNN.

4) **Normalization**: Each spectrogram was normalized by subtracting the mean and dividing by the standard deviation to improve training stability.

5) **Data Splitting**: The dataset was split into training (70%), validation (15%), and testing (15%) sets. To prevent data leakage, all segments from the same audio file were placed in the same split.

### C. Model Architecture

We implemented a convolutional neural network (CNN) architecture specifically designed for spectrogram image classification. The network consists of four convolutional blocks followed by fully connected layers:

1) **First Convolutional Block**:
   - 32 filters with 3×3 kernel size
   - ReLU activation
   - Batch normalization
   - 2×2 max pooling
   - Dropout (0.2)

2) **Second Convolutional Block**:
   - 64 filters with 3×3 kernel size
   - ReLU activation
   - Batch normalization
   - 2×2 max pooling
   - Dropout (0.3)

3) **Third Convolutional Block**:
   - 128 filters with 3×3 kernel size
   - ReLU activation
   - Batch normalization
   - 2×2 max pooling
   - Dropout (0.3)

4) **Fourth Convolutional Block**:
   - 256 filters with 3×3 kernel size
   - ReLU activation
   - Batch normalization
   - 2×2 max pooling
   - Dropout (0.4)

5) **Fully Connected Layers**:
   - Flattening layer
   - Dense layer with 512 units and ReLU activation
   - Batch normalization
   - Dropout (0.5)
   - Output layer with 10 units (one per genre)

### D. Training Procedure

The model was trained with the following configuration:
- Loss function: Cross-entropy loss
- Optimizer: Adam with initial learning rate of 0.001
- Batch size: 32
- Maximum epochs: 50
- Early stopping: Based on validation loss with patience of 5 epochs
- Learning rate scheduler: ReduceLROnPlateau with factor 0.5

During training, we monitored the loss and accuracy on both training and validation sets. We saved the model checkpoint with the best validation accuracy for final evaluation.

## III. Results and Analysis

### A. Model Performance

The trained CNN model achieved an overall accuracy of 71.35% on the test set. Table I summarizes the precision, recall, and F1-score for each genre class.

**TABLE I: CLASSIFICATION PERFORMANCE BY GENRE**

| Genre     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Blues     | 0.61      | 0.68   | 0.64     | 72      |
| Classical | 0.86      | 0.99   | 0.92     | 72      |
| Country   | 0.60      | 0.68   | 0.64     | 72      |
| Disco     | 0.82      | 0.62   | 0.71     | 72      |
| Hip-hop   | 0.73      | 0.82   | 0.77     | 72      |
| Jazz      | 0.84      | 0.86   | 0.85     | 78      |
| Metal     | 0.97      | 0.82   | 0.89     | 72      |
| Pop       | 0.74      | 0.68   | 0.71     | 72      |
| Reggae    | 0.63      | 0.61   | 0.62     | 72      |
| Rock      | 0.38      | 0.36   | 0.37     | 72      |
| **Avg/Total** | **0.72** | **0.71** | **0.71** | **726** |

### B. Discussion

As shown in Table I, the model performed exceptionally well on some genres while struggling with others. Key observations include:

1) **High-Performance Genres**: Classical (F1: 0.92), Metal (F1: 0.89), and Jazz (F1: 0.85) were recognized with high accuracy. These genres typically have distinctive spectral characteristics — classical music often features orchestral instruments with specific frequency patterns, metal is characterized by high energy across frequency bands, and jazz has recognizable rhythmic and harmonic structures.

2) **Low-Performance Genres**: Rock (F1: 0.37) had the lowest performance metrics. This might be attributed to its similarity with other genres like blues and metal, leading to significant confusion.

3) **Confusion Patterns**: Analysis of misclassifications revealed particular confusion between:
   - Rock and blues (stylistic similarities)
   - Country and pop (production techniques overlap)
   - Reggae and hip-hop (rhythmic patterns)

The overall accuracy of 71.35% is competitive with other published results in the field, especially considering the complexity of distinguishing between 10 different genres with overlapping characteristics.

## IV. Conclusion and Future Work

We developed a CNN-based system for automatic music genre classification using mel-spectrogram features. The model achieved promising results with an overall accuracy of 71.35% across ten genres. Performance varied significantly between genres, with classical, metal, and jazz being the most accurately classified.

Future improvements could include:

1) **Data Augmentation**: Implementing techniques such as pitch shifting, time stretching, and adding noise to increase model robustness.

2) **Architecture Enhancements**: Exploring recurrent layers (LSTM/GRU) to better capture temporal dependencies in music, or attention mechanisms to focus on discriminative parts of spectrograms.

3) **Ensemble Methods**: Combining multiple models or feature sets to improve classification accuracy.

4) **Multi-label Classification**: Acknowledging that many songs contain elements of multiple genres and implementing a multi-label approach.

## References

[1] T. Li, M. Ogihara, and Q. Li, "A comparative study on content-based music genre classification," in *Proc. SIGIR*, 2003, pp. 282-289.

[2] J. Lee, J. Park, K. Kim, and J. Nam, "Sample-level deep convolutional neural networks for music auto-tagging using raw waveforms," in *Proc. SMC*, 2017.

[3] S. Dieleman and B. Schrauwen, "End-to-end learning for music audio," in *IEEE ICASSP*, 2014, pp. 6964-6968.

[4] K. Choi, G. Fazekas, M. Sandler, and K. Cho, "Convolutional recurrent neural networks for music classification," in *IEEE ICASSP*, 2017, pp. 2392-2396.

[5] Z. Fu, G. Lu, K. M. Ting, and D. Zhang, "A survey of audio-based music classification and annotation," *IEEE Trans. Multimedia*, vol. 13, no. 2, pp. 303-319, 2011.