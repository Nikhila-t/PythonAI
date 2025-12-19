from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score,precision_recall_fscore_support, confusion_matrix, f1_score, precision_score, recall_score

from typing import Any, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

## Location where we save our trained model
MODEL_PATH = Path("models/customertickets_model.pkl")

## Performs Pipeline which internally handled Feature engineering using TF-IDF vectorisation
    ## Convert text to TF-IDF vectors (no embeddings)
    ## Encode categorical fields (product_module, customer_tier, priority)
    ## (No sentiment analysis nor NLP pipelines)
    ## Build pipeline with column transformer
## args df: cleaned DataFrame
## returns: Pipeline
def _build_pipeline() -> Pipeline:

    ## list the categorical fields (product_module, customer_tier, priority)
    categorical_cols = [
        "product_module",
        "customer_tier",
        "priority"
    ]

    ## Column transformer that acting as the pre-processor 
    ## to perform the TF-IDF vectorisation of combined_text while OneHotEncode the listed categorical cols
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=5000), "combined_text"),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["product_module", "customer_tier", "priority"])
        ]
    )

    ## Tried both RandomForestClassifier and Logistic Regression, the performance of Logistic Regressor is slightly better
    ## and hence chose Logistic Regressor for training our model

    ## classfier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier = LogisticRegression(max_iter=2000,class_weight="balanced",solver="lbfgs",n_jobs=-1,C=2.0)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline

## Loads the trained model from disk for reuse
## Raises file not found error if model file doesn't exist
## returns: model (aka pipeline)
def load_model() -> Pipeline:
    if not MODEL_PATH.exists:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}, train it first via /ml, /train-sample")
    
    model:Pipeline = joblib.load(MODEL_PATH)
    return model

## Train the Logistic regression model on the given dataFrame
## Model is evaluated and metrics are generated
## args: cleaned DataFrame originated from raw csv
## returns: metrics in Dict[str, Any] format
def train_model(df:pd.DataFrame) -> Dict[str, Any]:

    ## Listed columns that are considered as features (text and categorical) for this model
    feature_cols = [
        "combined_text",
        "product_module",
        "customer_tier",
        "priority"
    ]

    X = df[feature_cols]

    ## target column for predicting the team to be assigned
    y = df["assigned_team"]

    ## Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    model = _build_pipeline()

    ## fit = learn from data
    ## X_train -> Fits the pre-processor (learn TF-IDF for combined text features, categories for OneHotEncoding)
    ## y_train -> Fits the Logistic regressor on transformed features and labels
    model.fit(X_train, y_train)

    ## Evaluation
    ## y_pred uses the trained pipeline to predict assigned_team for the unseen X_test data
    ## preprocessor transforms X_test
    ## LogicticRegressor helps predict the assigned_team
    ## y_pred represents the predicted assigned_team
    y_pred = model.predict(X_test)

    ## Metrics
    ## Accuracy, Precision (per class), Recall (per class), F1-score and Confusion matrix

    ## Accuracy: Overall percentage of correct predictions
    accuracy = accuracy_score(y_test, y_pred)

    ## Precision (per class): Proportion of predicted samples for a class that are actually correct
    ## Recall (per class): Proportion of actual samples of a class that are correctly identified
    ## F1-score (per class): Harmonic mean of precision and recall, balancing correctness and coverage
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)

    ## confusion matrix is a table showing how predictions match actual classes, highlighting misclassifications
    conf_matrix = confusion_matrix(y_test, y_pred)

    ## Map per-class precision & recall
    per_class_metrics = {
        cls: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(fs), 4)
        }
        for cls, p, r, fs in zip(model.classes_, precision, recall, f1)
    }

    ## save the model to disk per provided Model path

    ## parent = True -> creates the model directory under parent directory
    ## exist_ok = True -> dont crash if directory already exists
    MODEL_PATH.parent.mkdir(parents = True, exist_ok = True)

    ## joblib.dump -> serialises the entire pipeline to disk
    joblib.dump(model, MODEL_PATH)

    metrics = {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "model_path": str(MODEL_PATH),
        "accuracy": round(float(accuracy), 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_matrix.tolist(),
        "labels": model.classes_.tolist(),
    }

    print(metrics)

    return metrics

## Predict which team should be assigned to the single ticket
## args: payload - Dict with following keys: ticket_subject, ticket_body, product_module, customer_tier, priority
## returns: Predicted assigned_team as str and Confidence_ccore as float
def predict_team_to_assign(payload: Dict) -> Dict[str, Any]:

    model = load_model()

    # From the payload, create a single-row DataFrame with same columns used during the training
    df_input = pd.DataFrame(
        [
            {
                "combined_text": payload.get("ticket_subject", "") + " " + payload.get("ticket_body", ""),
                "product_module": payload.get("product_module", "Unknown"),
                "customer_tier": payload.get("customer_tier", "Unknown"),
                "priority": payload.get("priority", "Unknown")
            }
        ]
    )

    probs = model.predict_proba(df_input)[0]

    predicted_team = model.classes_[np.argmax(probs)]

    confidence_score = float(np.max(probs))
    
    return {
        "predicted_team" : predicted_team,
        "confidence" : round(confidence_score, 4)
    }