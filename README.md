# Task 2 – Predictive Analysis Using Machine Learning

This project is part of my **Data Analyst Internship at CodeTech**. The goal of Task 2 is to build a machine learning model to predict **product star ratings** based on review data using classification techniques.

---

## Task Objective

> Build a machine learning model (regression or classification) to predict outcomes based on a dataset.

## Problem Statement

We aim to predict the **star rating** (1 to 5) of Amazon-style product reviews using the **review text** and **product title** as input features. This helps in understanding user sentiment and automating review evaluation at scale.

## Tools & Libraries Used

- Python
- Pandas
- scikit-learn
- TfidfVectorizer
- OneHotEncoder
- RandomForestClassifier
- Jupyter Notebook

## Dataset

A synthetic dataset was created containing:
- `review_id`: Unique ID for each review
- `product_title`: Product name (e.g., Bluetooth Speaker, Smart Watch)
- `review_body`: Text of the review
- `star_rating`: Rating from 1 to 5
- `customer_id`, `review_date`: Additional metadata

> 📄 File: `sample_amazon_reviews.csv`

## ML Workflow

1. **Data Cleaning** – Removed null values
2. **Text Preprocessing** – Used `TfidfVectorizer` for `review_body`
3. **Feature Encoding** – One-hot encoded `product_title`
4. **Train-Test Split** – 80% training / 20% testing
5. **Model Training** – Used `RandomForestClassifier`
6. **Model Evaluation** – Evaluated with accuracy and classification report
