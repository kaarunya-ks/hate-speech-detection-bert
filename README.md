# Hate Speech Detection using BERT

##  Project Overview
In recent years, social media has become an indispensable medium for global communication, enabling people to express themselves freely and share their opinions. However, this increased accessibility has also brought a significant surge in hate speech, which can have profound negative impacts on individuals and communities.

This project presents a BERT-based deep learning approach for hate speech classification on Twitter. BERT (Bidirectional Encoder Representations from Transformers), developed by Google, is a state-of-the-art NLP model known for its superior contextual understanding of text. In this project, we fine-tune BERT on a labeled dataset of tweets to classify posts as either hate speech or non-hate speech.

---

## Objective
To build an automated, scalable hate speech detection model that classifies tweets accurately using deep learning techniques.

---

## 📂 Dataset
- **Source**: Twitter dataset with labeled tweets.
- **Classes**:
  - Class 0: Non-hate speech
  - Class 1: Hate speech
- Reduced the dataset to 5,000 tweets for efficient training and testing.

---

##  Methodology

### 1. Data Collection
- A labeled Twitter dataset was used, with binary class labels:
  - 0: Non-hate speech
  - 1: Hate speech

### 2. Data Preprocessing
- **Text Cleaning**:
  - Removed URLs, mentions, hashtags, and special characters.
  - Converted all text to lowercase.
- **Data Split**:
  - 80% for training, 20% for testing.

### 3. Tokenization
- Used **BERT Tokenizer** (`bert-base-uncased`):
  - Tokenized text into subword tokens
  - Applied padding/truncation to a max length of 512 tokens
  - Generated input IDs and attention masks for model input

### 4. Model Selection
- **Base Model**: BERT-base-uncased
- Model includes:
  - Pre-trained BERT layers
  - A classification head for binary output (hate vs non-hate)

### 5. Model Training
- **Training Details**:
  - Epochs: 3
  - Optimizer: AdamW
- Evaluation performed after each epoch using validation data.

### 6. Performance Evaluation
- **Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **Visualization**:
  - Confusion matrix using Seaborn
  - Metric curves over epochs

### 7. Model Testing
- Model evaluated on test set using the same tokenization pipeline.

### 8. Prediction & Inference
- Simple CLI setup to allow users to enter new tweets.
- Predictions generated using softmax probabilities on model logits.

---

## 📈 Results
- **Accuracy Achieved**: 92%
- **Evaluation**: Confusion Matrix, Precision, Recall, F1-score

---

##  How to Run
1. Clone the repository or open the notebook in Google Colab.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt


 ## Technologies Used
- Python  
- BERT (Transformers by HuggingFace)  
- PyTorch  
- Pandas, NumPy, Seaborn, Matplotlib  
- Google Colab  
