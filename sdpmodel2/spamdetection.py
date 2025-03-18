# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import numpy as np
import re

from sklearn.ensemble import StackingClassifier


class SpamDetection:
    def __init__(self):
        # load and preprocess data
        self.dataset1_path = 'sdpmodel2/spam.csv'
        self.dataset2_path = 'sdpmodel2/spam2.csv'
        self.dataset3_path = 'sdpmodel2/spam3.csv'
        
        self.load_data()
        self.preprocess_data()
        self.train_models()

    def load_data(self):
        # load dataset
        dataset1 = pd.read_csv(self.dataset1_path, encoding='latin-1')
        dataset2 = pd.read_csv(self.dataset2_path, encoding="utf-8")
        dataset3 = pd.read_csv(self.dataset3_path, encoding='latin-1')
        
        # process dataset1
        dataset1.dropna(how='any', inplace=True)
        dataset1.drop_duplicates(inplace=True)
        dataset1['Category'] = dataset1['Category'].map({'ham': 'Not Spam', 'spam': 'Spam'})
        
        # process dataset2
        dataset2.columns = dataset2.columns.str.strip().str.lower()
        dataset2[['subject', 'message', 'Category']] = dataset2[['subject', 'messages', 'category']].fillna('Unknown')
        dataset2['Message'] = dataset2['subject'] + " " + dataset2['message']
        dataset2.drop(columns=['subject', 'message'], inplace=True)
        dataset2.drop_duplicates(inplace=True)
        dataset2.dropna(subset=['Message', 'Category'], inplace=True)
        dataset2['Category'] = dataset2['Category'].map({'0': 'Not Spam', '1': 'Spam'})
        
        
        # process dataset3
        categorical_cols = ['Message ID', 'Subject', 'Message', 'Category', 'Date']
        dataset3[categorical_cols] = dataset3[categorical_cols].fillna('Unknown')
        dataset3.drop(columns=['Message ID', 'Subject', 'Date'], inplace=True)
        dataset3.drop_duplicates(inplace=True)
        dataset3['Category'] = dataset3['Category'].map({'ham': 'Not Spam', 'spam': 'Spam'})
        
        # combine datasets
        self.dataset = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)
        
        #filter job related data
        
        self.job_keywords = ['job', 'work', 'career', 'employment', 'position', 'hire', 'recruit', 'opportunity', 'vacancy', 'position']
        self.dataset = self.dataset[self.dataset['Message'].str.contains('|'.join(self.job_keywords), case=False, na=False)]
        
        # remove special characters
        self.dataset['Message'] = self.dataset['Message'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        # final cleanup
        self.dataset = self.dataset.dropna(subset=['Message', 'Category'])
        self.message = self.dataset['Message']
        self.category = self.dataset['Category']
        
    # def dataset_info(self):
    #     return self.dataset.shape

    def preprocess_data(self):
        # split data into training and testing sets
        self.message_train, self.message_test, self.category_train, self.category_test = train_test_split(
            self.message, self.category, test_size=0.2, random_state=42
        )

        # vectorize text data
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.message_train_vec = self.vectorizer.fit_transform(self.message_train)
        self.message_test_vec = self.vectorizer.transform(self.message_test)

    def train_models(self):
        # train individual models
        self.mn_model = MultinomialNB()
        self.lr_model = LogisticRegression(max_iter=500)
        self.bn_model = BernoulliNB()
        self.rf_model = RandomForestClassifier(n_estimators=10)

        self.mn_model.fit(self.message_train_vec, self.category_train)
        self.lr_model.fit(self.message_train_vec, self.category_train)
        self.bn_model.fit(self.message_train_vec, self.category_train)
        self.rf_model.fit(self.message_train_vec, self.category_train)

    def model1_accuracy(self):
        return self.mn_model.score(self.message_test_vec, self.category_test)

    def model2_accuracy(self):
        return self.lr_model.score(self.message_test_vec, self.category_test)

    def model3_accuracy(self):
        return self.bn_model.score(self.message_test_vec, self.category_test)
    
    def model4_accuracy(self):
        return self.rf_model.score(self.message_test_vec, self.category_test)

    estimator = [
        ('MultinomialNB', MultinomialNB()),
        ('BernoulliNB', BernoulliNB()),
        ('RandomForestClassifier', RandomForestClassifier())
    ]
    
    stack_model = StackingClassifier(
        estimators=estimator, 
        final_estimator=LogisticRegression()
    )
    
    




    def ensemble_accuracy(self):
    # get predictions from all models
        pred_mn = self.mn_model.predict(self.message_test_vec)
        pred_lr = self.lr_model.predict(self.message_test_vec)
        pred_bn = self.bn_model.predict(self.message_test_vec)
        pred_rf = self.rf_model.predict(self.message_test_vec)

        # calculate ensemble predictions using majority voting with np.unique
        predictions = np.array([pred_mn, pred_lr, pred_bn, pred_rf])
        ensemble_predictions = np.array([
            np.unique(predictions[:, i], return_counts=True)[0][
                np.argmax(np.unique(predictions[:, i], return_counts=True)[1])
            ]
            for i in range(predictions.shape[1])
        ])

        # calculate accuracy of ensemble model
        return accuracy_score(self.category_test, ensemble_predictions)

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