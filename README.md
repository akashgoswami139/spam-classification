# SMS Spam Classification

https://spam-classification-akash.streamlit.app/

A Machine Learning project that classifies SMS messages as **Ham** or **Spam** using supervised learning.

### 🚀 Live Demo

🔗 **Try the application:** https://spam-classification-akash.streamlit.app/

🔗 **GitHub Repository:** https://github.com/akashgoswami139/spam-classification

## Overview

This project demonstrates an end-to-end text classification workflow, including:

- Exploratory Data Analysis
- Data cleaning and duplicate handling
- Target encoding
- Train-test splitting
- TF-IDF feature extraction
- Model training
- Model comparison
- Model evaluation
- Model serialization using Joblib
- Streamlit-based prediction interface

## Dataset

The dataset contains SMS messages with two main columns:

- **Category** – Target variable (`ham` or `spam`)
- **Message** – SMS text used as the input feature

The target classes were encoded as:

- `ham` → 0
- `spam` → 1

The dataset is imbalanced, with significantly more legitimate messages than spam messages.

## Machine Learning Approach

The project follows this workflow:

**Data → EDA → Preprocessing → TF-IDF → Model Training → Evaluation → Best Model → Deployment**

Since the main feature is unstructured text, TF-IDF was used to convert SMS messages into numerical representations suitable for machine learning models.

## Models Compared

Three machine learning algorithms were evaluated:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (SVM)

## Model Performance

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 97% | 97% | 74% | 84% |
| Multinomial Naive Bayes | 96% | 100% | 64% | 78% |
| **Linear SVM** | **98%** | **96%** | **84%** | **90%** |

## Best Model

### Linear SVM

Linear SVM achieved the best overall performance among the evaluated models.

**Performance:**

- **Accuracy:** 98%
- **Spam Precision:** 96%
- **Spam Recall:** 84%
- **Spam F1-Score:** 90%

The SVM model was selected because it provided the strongest balance between precision and recall for the spam class.

## Evaluation

Because the dataset is imbalanced, model performance was evaluated using more than accuracy.

Important metrics used:

- **Precision** – Measures how many predicted spam messages were actually spam.
- **Recall** – Measures how many actual spam messages were successfully detected.
- **F1-Score** – Provides a balance between precision and recall.
- **Accuracy** – Measures overall classification correctness.
- **Confusion Matrix** – Used to understand classification errors.

## Model Saving

The trained SVM model and TF-IDF vectorizer were saved using Joblib so that the model can be reused without retraining.

The saved artifacts are used by the prediction pipeline for real-time inference.

## Streamlit Application

The project includes a Streamlit interface where users can enter an SMS message and receive a prediction.

### Application Flow

**SMS Message → TF-IDF Transformation → Linear SVM → Ham / Spam Prediction**

The interface also displays the SVM decision score associated with the prediction.

## Project Structure

- `spam_classification.ipynb` – Complete notebook containing EDA, preprocessing, feature extraction, model training and evaluation
- `pipeline.py` – Prediction pipeline for loading the trained artifacts and generating predictions
- `app.py` – Streamlit user interface
- `spam_svm_model.pkl` – Trained Linear SVM model
- `tfidf_vectorizer.pkl` – Trained TF-IDF vectorizer
- `requirements.txt` – Project dependencies
- `README.md` – Project documentation

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Joblib
- Streamlit
- Jupyter Notebook
- Git
- GitHub

## How to Run

Clone the repository:

**https://github.com/akashgoswami139/spam_classification**

Install the required dependencies from `requirements.txt` and launch the Streamlit application.

## Key Learning Outcomes

This project helped demonstrate:

- Text preprocessing for machine learning
- Exploratory Data Analysis
- Handling imbalanced classification data
- Feature extraction using TF-IDF
- Comparing different classification algorithms
- Evaluating models using precision, recall and F1-score
- Selecting a model based on the actual problem requirements
- Saving and reusing trained ML models
- Integrating a machine learning model into a Streamlit application

## Future Improvements

- TF-IDF hyperparameter tuning
- Cross-validation
- SVM hyperparameter optimization
- Additional text-based feature engineering
- Error analysis
- Model monitoring
- Automated testing
- Cloud deployment

## Author

**Akash Goswami**

GitHub:  
https://github.com/akashgoswami139

LinkedIn:  
https://www.linkedin.com/in/akashgoswami-/

X:  
https://x.com/akashgoswami144


