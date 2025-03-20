# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import numpy as np
import re
import os
import pickle


class SpamDetection:
    def __init__(self, use_cache=False):  # Set default to False to force retraining
        # load and preprocess data
        self.dataset1_path = 'sdpmodel2/spam.csv'
        self.dataset2_path = 'sdpmodel2/spam2.csv'
        self.dataset3_path = 'sdpmodel2/spam3.csv'
        
        # Path for model caching
        self.model_cache_path = 'sdpmodel2/model_cache.pkl'
        self.use_cache = use_cache
        
        # Check if cached models exist and load them if requested
        if self.use_cache and os.path.exists(self.model_cache_path):
            self.load_cached_models()
        else:
            self.load_data()
            self.preprocess_data()
            self.train_models()
            if self.use_cache:
                self.cache_models()

    def load_cached_models(self):
        """Load models from cache to avoid retraining"""
        try:
            with open(self.model_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.vectorizer = cache_data['vectorizer']
                self.mn_model = cache_data['mn_model']
                self.lr_model = cache_data['lr_model']
                self.bn_model = cache_data['bn_model']
                self.rf_model = cache_data['rf_model']
                self.stack_model = cache_data['stack_model']
                self.bagging_model = cache_data['bagging_model']  # Load the bagging model
                self.message_test_vec = cache_data['message_test_vec']
                self.category_test = cache_data['category_test']
        except (FileNotFoundError, KeyError, pickle.UnpicklingError):
            # If any error occurs during loading, fall back to training
            self.load_data()
            self.preprocess_data()
            self.train_models()

    def cache_models(self):
        """Save trained models to cache for faster loading next time"""
        cache_data = {
            'vectorizer': self.vectorizer,
            'mn_model': self.mn_model,
            'lr_model': self.lr_model,
            'bn_model': self.bn_model,
            'rf_model': self.rf_model,
            'stack_model': self.stack_model,
            'bagging_model': self.bagging_model,  # Add the bagging model
            'message_test_vec': self.message_test_vec,
            'category_test': self.category_test
        }
        with open(self.model_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

    def load_data(self):
        # load all datasets to ensure model diversity
        dataset1 = pd.read_csv(self.dataset1_path, encoding='latin-1')
        
        # process dataset1
        dataset1.dropna(how='any', inplace=True)
        dataset1.drop_duplicates(inplace=True)
        dataset1['Category'] = dataset1['Category'].map({'ham': 'Not Spam', 'spam': 'Spam'})
        
        try:
            # load dataset2
            dataset2 = pd.read_csv(self.dataset2_path, encoding="utf-8")
            dataset2.columns = dataset2.columns.str.strip().str.lower()
            dataset2[['subject', 'message', 'Category']] = dataset2[['subject', 'messages', 'category']].fillna('Unknown')
            dataset2['Message'] = dataset2['subject'] + " " + dataset2['message']
            dataset2.drop(columns=['subject', 'message'], inplace=True)
            dataset2.drop_duplicates(inplace=True)
            dataset2.dropna(subset=['Message', 'Category'], inplace=True)
            dataset2['Category'] = dataset2['Category'].map({'0': 'Not Spam', '1': 'Spam'})
            
            # load dataset3
            dataset3 = pd.read_csv(self.dataset3_path, encoding='latin-1')
            categorical_cols = ['Message ID', 'Subject', 'Message', 'Category', 'Date']
            dataset3[categorical_cols] = dataset3[categorical_cols].fillna('Unknown')
            dataset3.drop(columns=['Message ID', 'Subject', 'Date'], inplace=True)
            dataset3.drop_duplicates(inplace=True)
            dataset3['Category'] = dataset3['Category'].map({'ham': 'Not Spam', 'spam': 'Spam'})
            
            # combine datasets
            self.dataset = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)
        except Exception as e:
            # If there's an error loading dataset2 or dataset3, just use dataset1
            print(f"Error loading additional datasets: {e}")
            self.dataset = dataset1
        
        # Use the full set of job keywords for better diversity
        self.job_keywords = ['job', 'work', 'career', 'employment', 'position', 'hire', 'recruit', 'opportunity', 'vacancy', 'position']
        self.dataset = self.dataset[self.dataset['Message'].str.contains('|'.join(self.job_keywords), case=False, na=False)]
        
        # Remove special characters
        self.dataset['Message'] = self.dataset['Message'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
        
        # final cleanup
        self.dataset = self.dataset.dropna(subset=['Message', 'Category'])
        self.message = self.dataset['Message']
        self.category = self.dataset['Category']

    def preprocess_data(self):
        # split data into training and testing sets
        self.message_train, self.message_test, self.category_train, self.category_test = train_test_split(
            self.message, self.category, test_size=0.2, random_state=42
        )

        # Use TfidfVectorizer without max_features to preserve model diversity
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.message_train_vec = self.vectorizer.fit_transform(self.message_train)
        self.message_test_vec = self.vectorizer.transform(self.message_test)

    def train_models(self):
        # Use different parameters for each model to ensure diversity
        self.mn_model = MultinomialNB(alpha=1.0)  # Default alpha
        self.lr_model = LogisticRegression(max_iter=500, C=1.0)  # Default C
        self.bn_model = BernoulliNB(alpha=0.5)  # Different alpha
        self.rf_model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Train models
        self.mn_model.fit(self.message_train_vec, self.category_train)
        self.lr_model.fit(self.message_train_vec, self.category_train)
        self.bn_model.fit(self.message_train_vec, self.category_train)
        self.rf_model.fit(self.message_train_vec, self.category_train)
        
        # Create stacking model
        self.stack_model = self.new_stacking_model()
        
        # Create bagging model - NEW CODE
        self.bagging_model = self.new_bagging_model()

    def model1_accuracy(self):
        return self.mn_model.score(self.message_test_vec, self.category_test)

    def model2_accuracy(self):
        return self.lr_model.score(self.message_test_vec, self.category_test)

    def model3_accuracy(self):
        return self.bn_model.score(self.message_test_vec, self.category_test)
    
    def model4_accuracy(self):
        return self.rf_model.score(self.message_test_vec, self.category_test)
    
    def new_stacking_model(self):
        # Use all models in the stacking classifier for better diversity
        estimator = [
            ('MultinomialNB', MultinomialNB(alpha=0.7)),  # Different alpha
            ('BernoulliNB', BernoulliNB(alpha=0.3)),  # Different alpha
            ('RandomForestClassifier', RandomForestClassifier(n_estimators=5, max_depth=5))  # Smaller forest
        ]
        
        stack_model = StackingClassifier(
            estimators=estimator, 
            final_estimator=LogisticRegression(max_iter=300, C=0.8),  # Different parameters
        )
        stack_model.fit(self.message_train_vec, self.category_train)
        return stack_model

    def stack_model_accuracy(self):
        # Get predictions from stacking model
        pred_sm = self.stack_model.predict(self.message_test_vec)
        # Calculate accuracy of ensemble model
        stack_accuracy_score = accuracy_score(self.category_test, pred_sm)
        return stack_accuracy_score
    
    def new_bagging_model(self):
        """Create and train a bagging ensemble model"""
        # Create base estimators for bagging
        base_nb = LogisticRegression(max_iter=500, C=1.0)
        
        # Create bagging classifier with the base estimator
        bagging_model = BaggingClassifier(
            estimator=base_nb,
            n_estimators=10,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=False,
            random_state=42
        )
        
        # Train the bagging model
        bagging_model.fit(self.message_train_vec, self.category_train)
        return bagging_model
    
    def bagging_model_accuracy(self):
        """Calculate accuracy of the bagging model"""
        pred_bg = self.bagging_model.predict(self.message_test_vec)
        bagging_accuracy_score = accuracy_score(self.category_test, pred_bg)
        return bagging_accuracy_score

    def predict(self, message):
        message_vec = self.vectorizer.transform([message])
        
        pred_mn = self.mn_model.predict(message_vec)[0]
        pred_lr = self.lr_model.predict(message_vec)[0]
        pred_bn = self.bn_model.predict(message_vec)[0]
        pred_rf = self.rf_model.predict(message_vec)[0]

        predictions = np.array([pred_mn, pred_lr, pred_bn, pred_rf])
        values, counts = np.unique(predictions, return_counts=True)
        final_prediction = values[np.argmax(counts)]
        
        return {
            'multinomial_nb': pred_mn,
            'logistic_regression': pred_lr,
            'bernoulli_nb': pred_bn,
            'random_forest': pred_rf,
            'ensemble': final_prediction
        }
    
    def stack_predict(self, message):
        message_vec = self.vectorizer.transform([message])
        pred_sm = self.stack_model.predict(message_vec)[0]
        return {
            'stack model': pred_sm,
            'ensemble': pred_sm
        }
        
    def bagging_predict(self, message):
        """Make predictions using the bagging model"""
        message_vec = self.vectorizer.transform([message])
        pred_bg = self.bagging_model.predict(message_vec)[0]
        return {
            'bagging model': pred_bg,
            'ensemble': pred_bg
        }
        
    def predict_all(self, message):
        """Make predictions using all models including both ensemble methods"""
        message_vec = self.vectorizer.transform([message])
        
        # Individual model predictions
        pred_mn = self.mn_model.predict(message_vec)[0]
        pred_lr = self.lr_model.predict(message_vec)[0]
        pred_bn = self.bn_model.predict(message_vec)[0]
        pred_rf = self.rf_model.predict(message_vec)[0]
        
        # Ensemble model predictions
        pred_stack = self.stack_model.predict(message_vec)[0]
        pred_bagging = self.bagging_model.predict(message_vec)[0]
        
        # Combine all predictions for a final ensemble
        predictions = np.array([pred_mn, pred_lr, pred_bn, pred_rf, pred_stack, pred_bagging])
        values, counts = np.unique(predictions, return_counts=True)
        final_prediction = values[np.argmax(counts)]
        
        return {
            'multinomial_nb': pred_mn,
            'logistic_regression': pred_lr,
            'bernoulli_nb': pred_bn,
            'random_forest': pred_rf,
            'stack_model': pred_stack,
            'bagging_model': pred_bagging,
            'ensemble': final_prediction
        }
