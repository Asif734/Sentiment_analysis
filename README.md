## Project Overview
This project aims to build a machine learning model for classifying sentence relationships into three categories: **Contradiction**, **Entailment**, and **Neutral**. The task involves exploratory data analysis (EDA), text preprocessing, model creation, evaluation, and optimization. The dataset consists of sentence pairs, and the goal is to predict the relationship between them.

## Dataset Description
The dataset contains pairs of sentences and their associated labels:
- **Contradiction**: The sentences contradict each other.
- **Entailment**: One sentence entails the other.
- **Neutral**: The sentences are unrelated or have no specific relationship.

## Steps

### Step 1: Exploratory Data Analysis (EDA)
Objective: Analyze the dataset to understand class distribution and text patterns.

#### Tasks:
- Visualize the distribution of Contradiction, Entailment, and Neutral labels.
- Analyze sentence structure, including sentence length, word distribution, and common words.
- Check for missing values or outliers.

### Step 2: Text Preprocessing
Objective: Clean and transform text for model training.

#### Tasks:
- **Tokenization**: Split sentences into words.
- **Lowercasing**: Convert text to lowercase.
- **Remove stop words, special characters, and punctuation.**
- **Stemming/Lemmatization**: Normalize words to their root form.
- **Feature Extraction**: Convert text into numeric representations using:
  - TF-IDF
  - Word2Vec
  - Transformer embeddings (e.g., BERT, XLM-R)

### Step 3: Model Creation
Objective: Train a machine learning model to classify sentence relationships.

#### Tasks:
- **Baseline Models**: Random Forest, Decision Trees, XGBoost (XGB).
- **Neural Networks**: Implement a Custom Artificial Neural Network (ANN).
- **Advanced Models**: Train LSTM/GRU models for sequence-based learning.
- **Transformer-Based Models**: Fine-tune BERT/XLM-R for contextual understanding.

### Step 4: Model Evaluation
Objective: Measure model performance using classification metrics.

#### Tasks:
- Compute **accuracy**, **precision**, **recall**, and **F1-score**.
- Plot a **Confusion Matrix** to analyze misclassifications.
- Generate an **AUC-ROC curve** to evaluate classification performance.

### Step 5: Model Tuning and Optimization
Objective: Improve model performance through tuning.


