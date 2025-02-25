# import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 

# first algorithm
from sklearn.naive_bayes import MultinomialNB

# second algorithm
from sklearn.linear_model import LogisticRegression


# for interface
import streamlit as st

# import data
data = pd.read_csv("D:\Research\Spam-Detection-ML\sdpmodel2\spam.csv", encoding='latin-1')



# data analizing 

data.drop_duplicates(inplace=True)

data['Category'] = data['Category'].map({'ham':'Not Spam', 'spam':'Spam'})
message = data['Message']
category = data['Category']

# data1 = data.dropna();

message_train, message_test, category_train, category_test = train_test_split(message, category, test_size=0.2)

cvData = CountVectorizer(stop_words='english')
cvTransform = cvData.fit_transform(message_train)


# train model

mnModel = MultinomialNB()
mnModel.fit(cvTransform, category_train)

lrModel = LogisticRegression()
lrModel.fit(cvTransform, category_train)

# test model
cvDataTest = cvData.transform(message_test)

# data predict

def model1Accuracy():
    return mnModel.score()

def predict(rt_message):
    atf_message = cvData.transform([rt_message]).toarray()
    prediction_nb = mnModel.predict(atf_message)
    prediction_lr = lrModel.predict(atf_message)
    
    # Return both predictions as a tuple
    return {
        'naive_bayes': prediction_nb[0],
        'logistic_regression': prediction_lr[0],
        'consensus': 'Spam' if (prediction_nb[0] == 'Spam' and prediction_lr[0] == 'Spam') else 'Not Spam'
    }






# st.header('Spam Detection')

# rt_message = st.text_input('Enter your message', key='message')

# st.write('The message is: ', rt_message)

# if st.button('Message Validate') :
#     if rt_message == '':
#         st.write('Please enter your message')
#         st.stop()
#     if predict(rt_message) == 'Spam':
#         st.write('This message is spam')
#     else:
#         st.write('This message is not spam')