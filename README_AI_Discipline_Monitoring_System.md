# AI-Based Discipline Monitoring System

## Overview

The AI-Based Discipline Monitoring System is a deep learning and computer vision project designed to automate the monitoring of student discipline in educational institutions. The system analyzes images and identifies whether students comply with institutional dress code and grooming regulations. It detects multiple discipline-related attributes, including dress type, beard presence, shirt tuck-in status, and ID card wearing status.

The project uses Convolutional Neural Networks (CNNs) built with TensorFlow and Keras to perform image classification tasks. A user-friendly dashboard has also been developed to display detection results, compliance statistics, confidence scores, and recent monitoring records in real time.

## Features

- Automated discipline monitoring using AI and computer vision
- Formal vs Casual dress code detection
- Beard vs No Beard classification
- Tuck-in vs Untucked shirt detection
- ID Card Wearing vs Not Wearing detection
- Real-time image analysis
- Interactive dashboard for result visualization
- Confidence score display for predictions
- Scalable and modular architecture
- Performance evaluation using machine learning metrics

## Technologies Used

### Programming Language
- Python 3.8+

### Deep Learning Framework
- TensorFlow
- Keras

### Computer Vision
- OpenCV

### Data Processing
- NumPy
- Pandas

### Visualization
- Matplotlib
- Seaborn

### Machine Learning Utilities
- Scikit-learn

### Dashboard Development
- Streamlit

## System Architecture

The system follows a modular architecture consisting of:

1. Image Input Layer
2. Data Preprocessing Layer
3. Feature Extraction using CNN
4. Classification Modules
5. Decision Integration Layer
6. Dashboard Visualization Layer

The architecture allows independent training and deployment of each discipline-checking module while producing a unified compliance report.

## Dataset

A custom image dataset was created for training and testing the models. The dataset contains labeled images organized into the following categories:

### Dress Code
- Formal
- Casual

### Beard Detection
- Beard
- No Beard

### Tuck-in Detection
- Tuck-in
- Untucked

### ID Card Detection
- Wearing ID
- Not Wearing ID

Data augmentation techniques such as rotation, flipping, zooming, and brightness adjustment were applied to improve model performance and generalization.

## Project Workflow

### Step 1: Dataset Preparation
Images are collected, labeled, resized, normalized, and augmented for training.

### Step 2: Data Preprocessing
Images are converted into a suitable format and split into training, validation, and testing datasets.

### Step 3: Model Training
Separate CNN models are trained for each discipline parameter.

### Step 4: Multi-Attribute Detection
The trained models are integrated into a unified discipline monitoring framework.

### Step 5: Dashboard Visualization
Results are displayed through an interactive dashboard showing compliance metrics and prediction confidence.

### Step 6: Evaluation and Optimization
Models are evaluated using accuracy, precision, recall, and F1-score.

## Installation

```bash
git clone https://github.com/Udayakumar-G/AICTE-B1-MAY2026--IBM-INTERNSHIP.git
cd AI-Discipline-Monitoring-System
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
streamlit run app.py
```

For model training:

```bash
python discipline.py
```

For prediction:

```bash
python detection.py
```

## Project Structure

```text
AI-Discipline-Monitoring-System/
│
├── dataset/
├── models/
├── dashboard/
├── training/
├── prediction/
├── requirements.txt
└── README.md
```

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Dashboard

The integrated dashboard provides:
- Total detections
- Discipline compliance percentage
- Confidence scores
- Recent detection history
- Visual analytics
- Real-time monitoring interface

## Future Enhancements

- Live CCTV integration
- Face recognition for attendance
- Student identity verification
- Mobile application support
- Cloud deployment
- Multi-person detection
- Behavioral analytics
- Violation reporting system

## Applications

- Educational Institutions
- Schools and Colleges
- Corporate Offices
- Training Centers
- Laboratories
- Security and Compliance Monitoring Systems

## Conclusion

The AI-Based Discipline Monitoring System demonstrates the practical application of artificial intelligence and computer vision in automating discipline monitoring. By combining deep learning models with an interactive dashboard, the system provides an accurate, transparent, scalable, and efficient solution for evaluating dress code and grooming compliance in educational institutions.

