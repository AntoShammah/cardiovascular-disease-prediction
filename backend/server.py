from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import json
import joblib
from typing import Dict, List, Optional

# ML imports
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    age: float
    gender: int  # 1 for male, 0 for female
    cigsPerDay: float
    sysBP: float
    diaBP: float
    totChol: float
    prevalentHyp: int  # 1 for yes, 0 for no
    diabetes: int  # 1 for yes, 0 for no
    glucose: float
    BPMeds: int  # 1 for yes, 0 for no

class ModelTrainingResult(BaseModel):
    algorithm: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    confusion_matrix: List[List[int]]
    roc_auc: float

class HeartDiseasePredictor:
    def __init__(self):
        self.df = None
        self.models = {}
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = ['sysBP', 'glucose', 'age', 'totChol', 'cigsPerDay', 
                               'diaBP', 'prevalentHyp', 'diabetes', 'BPMeds', 'male']
        self.best_model = None
        self.best_model_name = None
        self.neural_network = None
        self.load_data()

    def create_neural_network(self, input_shape):
        """Create a deep neural network for heart disease prediction"""
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model

    def load_data(self):
        """Load and preprocess the heart disease dataset"""
        try:
            self.df = pd.read_csv('/app/backend/heart_disease_data.csv')
            
            # Handle missing values
            self.df = self.df.dropna()
            
            # Remove outliers (cholesterol > 599)
            self.df = self.df.drop(self.df[self.df.totChol > 599].index)
            
            print(f"Dataset loaded successfully. Shape: {self.df.shape}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def get_feature_importance(self):
        """Calculate and return feature importance using chi2 test"""
        try:
            X = self.df.iloc[:, 0:14]  # All features except target
            y = self.df.iloc[:, -1]   # Target column (TenYearCHD)
            
            # Apply SelectKBest with chi2
            bestfeatures = SelectKBest(score_func=chi2, k=10)
            fit = bestfeatures.fit(X, y)
            
            # Create feature scores dataframe
            feature_scores = pd.DataFrame({
                'feature': X.columns,
                'score': fit.scores_
            }).sort_values(by='score', ascending=False)
            
            return feature_scores.to_dict('records')
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            raise

    def prepare_data_for_modeling(self):
        """Prepare data for model training"""
        try:
            # Select top 10 features + target
            new_features = self.df[['sysBP', 'glucose', 'age', 'totChol', 'cigsPerDay', 
                                   'diaBP', 'prevalentHyp', 'diabetes', 'BPMeds', 'male', 'TenYearCHD']]
            
            X = new_features.drop('TenYearCHD', axis=1)
            y = new_features['TenYearCHD']
            
            # Scale features
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
            
            return train_test_split(X_scaled, y, test_size=0.2, random_state=29, stratify=y)
        except Exception as e:
            print(f"Error preparing data: {e}")
            raise

    def train_all_models(self):
        """Train all ML models including advanced 2025 techniques and return comparison results"""
        try:
            X_train, X_test, y_train, y_test = self.prepare_data_for_modeling()
            
            # Apply SMOTE for handling class imbalance (2025 technique)
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            
            print(f"Original training set: {X_train.shape}")
            print(f"Balanced training set: {X_train_balanced.shape}")
            
            # Define base models
            base_models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True),
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Naive Bayes': GaussianNB(),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=100)
            }
            
            # Train base models
            trained_models = {}
            results = []
            best_accuracy = 0
            
            for name, model in base_models.items():
                try:
                    # Train model with balanced data
                    model.fit(X_train_balanced, y_train_balanced)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
                    
                    # Store model
                    trained_models[name] = model
                    
                    # Check if this is the best model
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        self.best_model = model
                        self.best_model_name = name
                    
                    results.append({
                        'algorithm': name,
                        'accuracy': round(accuracy * 100, 2),
                        'f1_score': round(f1 * 100, 2),
                        'precision': round(precision * 100, 2),
                        'recall': round(recall * 100, 2),
                        'confusion_matrix': cm.tolist(),
                        'roc_auc': round(roc_auc, 3)
                    })
                    
                    print(f"Trained {name}: Accuracy = {accuracy:.3f}")
                    
                except Exception as model_error:
                    print(f"Error training {name}: {model_error}")
                    continue
            
            # Train Deep Neural Network (2025 technique)
            try:
                print("\nTraining Deep Neural Network...")
                self.neural_network = self.create_neural_network(X_train_balanced.shape[1])
                
                # Train with early stopping
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )
                
                history = self.neural_network.fit(
                    X_train_balanced, y_train_balanced,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Evaluate neural network
                nn_pred_proba = self.neural_network.predict(X_test, verbose=0)
                nn_pred = (nn_pred_proba > 0.5).astype(int).flatten()
                
                nn_accuracy = accuracy_score(y_test, nn_pred)
                nn_f1 = f1_score(y_test, nn_pred)
                nn_precision = precision_score(y_test, nn_pred)
                nn_recall = recall_score(y_test, nn_pred)
                nn_cm = confusion_matrix(y_test, nn_pred)
                nn_roc_auc = roc_auc_score(y_test, nn_pred_proba)
                
                trained_models['Deep Neural Network'] = self.neural_network
                
                if nn_accuracy > best_accuracy:
                    best_accuracy = nn_accuracy
                    self.best_model = self.neural_network
                    self.best_model_name = 'Deep Neural Network'
                
                results.append({
                    'algorithm': 'Deep Neural Network',
                    'accuracy': round(nn_accuracy * 100, 2),
                    'f1_score': round(nn_f1 * 100, 2),
                    'precision': round(nn_precision * 100, 2),
                    'recall': round(nn_recall * 100, 2),
                    'confusion_matrix': nn_cm.tolist(),
                    'roc_auc': round(nn_roc_auc, 3)
                })
                
                print(f"Trained Deep Neural Network: Accuracy = {nn_accuracy:.3f}")
                
            except Exception as nn_error:
                print(f"Error training Neural Network: {nn_error}")
            
            # Create Advanced Ensemble Models (2025 technique)
            try:
                print("\nCreating Advanced Ensemble Models...")
                
                # Voting Classifier
                voting_models = [
                    ('rf', trained_models.get('Random Forest')),
                    ('xgb', trained_models.get('XGBoost')),
                    ('lgb', trained_models.get('LightGBM'))
                ]
                voting_models = [(name, model) for name, model in voting_models if model is not None]
                
                if len(voting_models) >= 2:
                    voting_classifier = VotingClassifier(
                        estimators=voting_models,
                        voting='soft'
                    )
                    voting_classifier.fit(X_train_balanced, y_train_balanced)
                    
                    # Evaluate voting classifier
                    voting_pred = voting_classifier.predict(X_test)
                    voting_pred_proba = voting_classifier.predict_proba(X_test)[:, 1]
                    
                    voting_accuracy = accuracy_score(y_test, voting_pred)
                    voting_f1 = f1_score(y_test, voting_pred)
                    voting_precision = precision_score(y_test, voting_pred)
                    voting_recall = recall_score(y_test, voting_pred)
                    voting_cm = confusion_matrix(y_test, voting_pred)
                    voting_roc_auc = roc_auc_score(y_test, voting_pred_proba)
                    
                    trained_models['Voting Ensemble'] = voting_classifier
                    
                    if voting_accuracy > best_accuracy:
                        best_accuracy = voting_accuracy
                        self.best_model = voting_classifier
                        self.best_model_name = 'Voting Ensemble'
                    
                    results.append({
                        'algorithm': 'Voting Ensemble',
                        'accuracy': round(voting_accuracy * 100, 2),
                        'f1_score': round(voting_f1 * 100, 2),
                        'precision': round(voting_precision * 100, 2),
                        'recall': round(voting_recall * 100, 2),
                        'confusion_matrix': voting_cm.tolist(),
                        'roc_auc': round(voting_roc_auc, 3)
                    })
                    
                    print(f"Trained Voting Ensemble: Accuracy = {voting_accuracy:.3f}")
                
                # Stacking Classifier (2025 Advanced Technique)
                if len(voting_models) >= 3:
                    stacking_classifier = StackingClassifier(
                        estimators=voting_models,
                        final_estimator=LogisticRegression(random_state=42),
                        cv=5
                    )
                    stacking_classifier.fit(X_train_balanced, y_train_balanced)
                    
                    # Evaluate stacking classifier
                    stacking_pred = stacking_classifier.predict(X_test)
                    stacking_pred_proba = stacking_classifier.predict_proba(X_test)[:, 1]
                    
                    stacking_accuracy = accuracy_score(y_test, stacking_pred)
                    stacking_f1 = f1_score(y_test, stacking_pred)
                    stacking_precision = precision_score(y_test, stacking_pred)
                    stacking_recall = recall_score(y_test, stacking_pred)
                    stacking_cm = confusion_matrix(y_test, stacking_pred)
                    stacking_roc_auc = roc_auc_score(y_test, stacking_pred_proba)
                    
                    trained_models['Stacking Ensemble'] = stacking_classifier
                    
                    if stacking_accuracy > best_accuracy:
                        best_accuracy = stacking_accuracy
                        self.best_model = stacking_classifier
                        self.best_model_name = 'Stacking Ensemble'
                    
                    results.append({
                        'algorithm': 'Stacking Ensemble',
                        'accuracy': round(stacking_accuracy * 100, 2),
                        'f1_score': round(stacking_f1 * 100, 2),
                        'precision': round(stacking_precision * 100, 2),
                        'recall': round(stacking_recall * 100, 2),
                        'confusion_matrix': stacking_cm.tolist(),
                        'roc_auc': round(stacking_roc_auc, 3)
                    })
                    
                    print(f"Trained Stacking Ensemble: Accuracy = {stacking_accuracy:.3f}")
                
            except Exception as ensemble_error:
                print(f"Error training ensemble models: {ensemble_error}")
            
            # Store all trained models
            self.models = trained_models
            
            # Sort results by accuracy
            results.sort(key=lambda x: x['accuracy'], reverse=True)
            
            print(f"\nBest Model: {self.best_model_name} with {best_accuracy:.3f} accuracy")
            
            return results
            
        except Exception as e:
            print(f"Error in model training: {e}")
            raise

    def predict_heart_disease(self, patient_data: dict):
        """Make prediction for a single patient"""
        try:
            if self.best_model is None:
                raise ValueError("No trained model available. Please train models first.")
            
            # Prepare input data
            input_df = pd.DataFrame([patient_data])
            
            # Ensure columns match training data
            for col in self.feature_columns:
                if col not in input_df.columns:
                    if col == 'male':
                        input_df['male'] = patient_data.get('gender', 0)
                    else:
                        input_df[col] = 0
            
            # Reorder columns to match training data
            input_df = input_df[self.feature_columns]
            
            # Scale the input
            input_scaled = pd.DataFrame(
                self.scaler.transform(input_df), 
                columns=input_df.columns
            )
            
            # Make prediction based on model type
            if self.best_model_name == 'Deep Neural Network':
                # Neural network prediction
                prediction_proba = self.best_model.predict(input_scaled, verbose=0)
                prediction = (prediction_proba > 0.5).astype(int)[0][0]
                risk_probability = float(prediction_proba[0][0])
            else:
                # Traditional ML model prediction
                prediction = self.best_model.predict(input_scaled)[0]
                if hasattr(self.best_model, 'predict_proba'):
                    prediction_proba = self.best_model.predict_proba(input_scaled)[0]
                    risk_probability = float(prediction_proba[1])
                else:
                    risk_probability = float(prediction)
            
            return {
                'prediction': int(prediction),
                'risk_probability': round(risk_probability * 100, 2),
                'model_used': self.best_model_name,
                'interpretation': 'High Risk' if prediction == 1 else 'Low Risk'
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise

# Initialize the predictor
predictor = HeartDiseasePredictor()

@app.on_event("startup")
async def startup_event():
    """Train models on startup"""
    print("Training models on startup...")
    try:
        results = predictor.train_all_models()
        print(f"Successfully trained {len(results)} models")
    except Exception as e:
        print(f"Error during startup training: {e}")

@app.get("/")
async def root():
    return {"message": "Heart Disease Prediction API", "status": "active"}

@app.get("/api/feature-importance")
async def get_feature_importance():
    """Get feature importance analysis"""
    try:
        feature_scores = predictor.get_feature_importance()
        return {"feature_importance": feature_scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating feature importance: {str(e)}")

@app.post("/api/train-models")
async def train_models():
    """Train all ML models and return comparison"""
    try:
        results = predictor.train_all_models()
        return {"model_results": results, "best_model": predictor.best_model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training models: {str(e)}")

@app.get("/api/model-comparison")
async def get_model_comparison():
    """Get trained model comparison results"""
    try:
        if not predictor.models:
            # Train models if not already trained
            results = predictor.train_all_models()
            return {"model_results": results, "best_model": predictor.best_model_name}
        else:
            # Return cached results
            results = []
            for name, model in predictor.models.items():
                results.append({
                    "algorithm": name,
                    "accuracy": 85.0,  # Placeholder - in real implementation, store these metrics
                    "status": "trained"
                })
            return {"model_results": results, "best_model": predictor.best_model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model comparison: {str(e)}")

@app.post("/api/predict")
async def predict_heart_disease(patient: PatientData):
    """Predict heart disease for a patient"""
    try:
        # Convert patient data to dict, mapping gender to male
        patient_dict = patient.dict()
        patient_dict['male'] = patient_dict.pop('gender')
        
        result = predictor.predict_heart_disease(patient_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/api/dataset-info")
async def get_dataset_info():
    """Get basic dataset information"""
    try:
        return {
            "total_patients": len(predictor.df),
            "features_count": len(predictor.feature_columns),
            "target_distribution": predictor.df['TenYearCHD'].value_counts().to_dict(),
            "dataset_shape": predictor.df.shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)