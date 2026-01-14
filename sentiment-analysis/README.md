# Sentiment Analysis using TF-IDF and Logistic Regression

This project implements a basic sentiment analysis pipeline for text data.

The goal is to classify text into sentiment categories using traditional machine learning techniques.

## Approach

1. Load and preprocess text data
2. Convert text into numerical features using TF-IDF
3. Train a Logistic Regression classifier
4. Evaluate model performance on test data

## Preprocessing Steps

- Convert text to lowercase
- Remove special characters
- Handle empty or missing text values
- Remove stopwords

## Feature Extraction

- TF-IDF (Term Frequency–Inverse Document Frequency) is used to represent text numerically
- Common words are down-weighted while important words receive higher weights

## Model Used

- Logistic Regression
- Chosen for simplicity and effectiveness in text classification tasks

## Evaluation

Model performance is evaluated using:
- Accuracy
- Precision, Recall, and F1-score (classification report)

## Files

- `sentiment_analysis.py` — Complete implementation of preprocessing, feature extraction, training, and evaluation

## Purpose

This project was developed as a learning exercise to understand the end-to-end workflow of a basic NLP classification problem.
