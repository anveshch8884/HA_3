🧠 Deep Learning Projects with Autoencoders and LSTM (MNIST & IMDB)
This project demonstrates three core applications of deep learning using TensorFlow/Keras:

Basic Autoencoder on MNIST dataset

Denoising Autoencoder for image denoising

LSTM Network for Sentiment Classification on IMDB reviews

📁 Contents
Q1: Train a basic autoencoder on MNIST

Q2: Train a denoising autoencoder on noisy MNIST digits and compare it with a basic autoencoder

Q3: Build an LSTM model for binary sentiment classification on IMDB dataset

🔸 Q1: Basic Autoencoder on MNIST
✅ Objective
Train an autoencoder to reconstruct images of handwritten digits from the MNIST dataset by learning a compressed (latent) representation.

📚 Dataset
MNIST – 28x28 grayscale images of digits (0-9)

⚙️ Steps
Load and normalize the MNIST dataset

Flatten images (784-dimensional vectors)

Define an encoder-decoder model using fully connected layers

Compile and train the model using binary_crossentropy loss

Visualize:

Training and validation loss

Reconstructed images vs original

Compare results across different latent dimensions (16, 32, 64, 128)

📐 Architecture
scss
Input (784) → Dense(latent_dim, relu) → Dense(784, sigmoid)
📊 Results
The autoencoder learns to compress and then reconstruct digit images.

Reconstruction quality improves with increased latent dimensions, but training time increases.

🔸 Q2: Denoising Autoencoder
✅ Objective
Improve the robustness of the autoencoder by training it to remove Gaussian noise from input images.

⚙️ Steps
Add Gaussian noise to MNIST images

Build a similar encoder-decoder model

Train on noisy inputs but with clean targets

Compare reconstruction results with a regular autoencoder

Evaluate both with MSE (Mean Squared Error) and visual comparisons

🧪 Experiment
Latent dimension: 32

Visualize:

Noisy input vs reconstructed output vs ground truth

Side-by-side comparison between Basic AE and Denoising AE

📊 Results
Denoising AE significantly outperforms Basic AE on noisy inputs.

Quantified using MSE.

🔸 Q3: LSTM for Sentiment Analysis on IMDB
✅ Objective
Build a deep learning model to classify movie reviews from the IMDB dataset as positive or negative.

📚 Dataset
IMDB – 50,000 movie reviews labeled as positive/negative

⚙️ Steps
Load and tokenize the dataset (limit vocab to top 10,000 words)

Pad sequences to fixed length (500)

Build a model with:

Embedding layer

LSTM layer with dropout

Dense sigmoid output

Train using binary_crossentropy loss

Evaluate:

Accuracy and loss plots

Confusion matrix

Precision/Recall trade-off at thresholds (0.3, 0.5, 0.7)

📐 Architecture
scss

Embedding(10000, 128) → LSTM(64) → Dense(1, sigmoid)
📊 Results
Achieves good validation accuracy

Visual analysis helps understand how threshold tuning affects precision and recall

🛠️ Installation
Make sure you have the following libraries installed:

pip install tensorflow numpy matplotlib seaborn scikit-learn
📸 Sample Visuals
Basic Autoencoder (MNIST)
Original vs Reconstructed Images

Denoising Autoencoder
Noisy vs Reconstructed vs Ground Truth

LSTM IMDB Model
Training Accuracy & Confusion Matrix
