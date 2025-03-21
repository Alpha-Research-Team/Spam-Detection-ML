from sdpmodel2.spamdetection import SpamDetection
import streamlit as st

spam_detection = SpamDetection()

def loadCss(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
# Page configuration
st.set_page_config(
    page_title="Spam Detection using ML",
    page_icon=":robot_face:",
    layout="wide",
)

loadCss('style/style.css')

with st.container():
    st.title("Spam Detection using ML")
    st.write("Instantly check if a message is spam using Machine Learning.")
 
 
with st.container():
    st.write("---")
    left_column, mid, right_column = st.columns((5,1,6))
    # Replace the existing validation button section with this code
with left_column:
    st.subheader("Enter your message:")
    rt_message = st.text_input("Text Only", placeholder="Type your message here...")
    if st.button('Validate 🚀'):
        if rt_message.strip() == '':
            st.warning('⚠️ Please enter a valid message.')
        else:
            # Use predict_all instead of stack_predict
            result = spam_detection.predict_all(rt_message)
            
            # Display the final ensemble result prominently
            if result['ensemble'] == 'Spam':
                st.markdown(
                    f"<div class='result-box spam'>❌ This message is <b>SPAM</b>.</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result-box not-spam'>✅ This message is <b>NOT SPAM</b>.</div>",
                    unsafe_allow_html=True
                )
            
            # Display individual model predictions
            st.write('---')
            st.subheader("Model Predictions")
            
            # Create three columns for better organization
            col1, col2, col3 = st.columns(3)
            
            # Basic models in first column
            with col1:
                st.markdown("<h4 style='text-align: center;'>Basic Models</h4>", unsafe_allow_html=True)
                
                # Multinomial NB
                if result['multinomial_nb'] == 'Spam':
                    st.markdown("<div class='model-result spam'>Multinomial NB: Spam</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='model-result not-spam'>Multinomial NB: Not Spam</div>", unsafe_allow_html=True)
                
                # Logistic Regression
                if result['logistic_regression'] == 'Spam':
                    st.markdown("<div class='model-result spam'>Logistic Regression: Spam</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='model-result not-spam'>Logistic Regression: Not Spam</div>", unsafe_allow_html=True)
            
            # More basic models in second column
            with col2:
                st.markdown("<h4 style='text-align: center;'>Basic Models</h4>", unsafe_allow_html=True)
                
                # Bernoulli NB
                if result['bernoulli_nb'] == 'Spam':
                    st.markdown("<div class='model-result spam'>Bernoulli NB: Spam</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='model-result not-spam'>Bernoulli NB: Not Spam</div>", unsafe_allow_html=True)
                
                # Random Forest
                if result['random_forest'] == 'Spam':
                    st.markdown("<div class='model-result spam'>Random Forest: Spam</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='model-result not-spam'>Random Forest: Not Spam</div>", unsafe_allow_html=True)
            
            # Ensemble models in third column
            with col3:
                st.markdown("<h4 style='text-align: center;'>Ensemble Models</h4>", unsafe_allow_html=True)
                
                # Stack Model
                if result['stack_model'] == 'Spam':
                    st.markdown("<div class='model-result spam'>Stack Model: Spam</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='model-result not-spam'>Stack Model: Not Spam</div>", unsafe_allow_html=True)
                
                # Bagging Model
                if result['bagging_model'] == 'Spam':
                    st.markdown("<div class='model-result spam'>Bagging Model: Spam</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='model-result not-spam'>Bagging Model: Not Spam</div>", unsafe_allow_html=True)
            
            # Voting information
            st.write('---')
            st.markdown("<h4>Voting Results</h4>", unsafe_allow_html=True)
            
            # Count votes
            spam_votes = sum(1 for key, value in result.items() if value == 'Spam' and key != 'ensemble')
            not_spam_votes = sum(1 for key, value in result.items() if value == 'Not Spam' and key != 'ensemble')
            total_votes = spam_votes + not_spam_votes
            
            # Display vote counts
            st.markdown(f"<div style='display: flex; justify-content: space-around;'>"
                        f"<div><b>Spam Votes:</b> {spam_votes}/{total_votes} ({spam_votes/total_votes*100:.1f}%)</div>"
                        f"<div><b>Not Spam Votes:</b> {not_spam_votes}/{total_votes} ({not_spam_votes/total_votes*100:.1f}%)</div>"
                        f"</div>", unsafe_allow_html=True)
            
            # Display vote distribution as a progress bar
            st.progress(spam_votes/total_votes)
            
            # Add a note about the ensemble decision
            st.markdown(f"<div style='text-align: center; margin-top: 10px;'>"
                        f"<i>The final decision is based on majority voting across all models.</i>"
                        f"</div>", unsafe_allow_html=True)

    
    
    with right_column:
        st.subheader("Our Models & Accuracy")
        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            nb_accuracy = (spam_detection.model1_accuracy() * 100)
            st.progress(nb_accuracy/100)
            st.write(f"Naive Bayes (Multinomial) : {nb_accuracy:.2f}%")
            
            # lr_accuracy = (spam_detection.model2_accuracy() * 100)
            # st.progress(lr_accuracy/100)
            # st.write(f"Logistic Regression : {lr_accuracy:.2f}%")
        
        with middle_column:
            bn_accuracy = (spam_detection.model3_accuracy() * 100)
            st.progress(bn_accuracy/100)
            st.write(f"Naive Bayes (Bernoulli) : {bn_accuracy:.2f}%")
            
           
            
    
        with right_column:
            rf_accuracy = (spam_detection.model4_accuracy() * 100)
            st.progress(rf_accuracy/100)
            st.write(f"Random Forest : {rf_accuracy:.2f}%")
            
            
            # bn_accuracy = (sd.model4Accuracy() * 100)
            # st.progress(bn_accuracy/100)
            # st.write(f"Naive Bayes (Bernoulli) - Accuracy: {bn_accuracy:.2f}%")
            
        nmf_accuracy = (spam_detection.stack_model_accuracy() * 100)
        bgg_accuracy = (spam_detection.bagging_model_accuracy() * 100)
        # dataset_size = spam_detection.dataset_info()
        with st.container():
            left_column, right_column = st.columns(2)
            with left_column:
                st.progress(nmf_accuracy/100)
                st.write(f"Stack Model - Accuracy: {nmf_accuracy:.2f}%")
            
            with right_column:
                st.progress(bgg_accuracy/100)
                st.write(f"Bagging Model - Accuracy: {bgg_accuracy:.2f}%")
            # st.write(f"Dataset Size: {dataset_size}")

# with st.container():
#     st.write("---")
#     left_column, middle_column, right_column = st.columns((3,1,2))
#     with left_column:
#         st.write("Re Train New Model")
#         if st.button("Re Train New Model"):
#             spam_detection.train_models()

with st.container():
    st.write("---")
    st.write("Team Members : Mr. Ushmika Kavishan - Miss. Rashmika Malshani - Miss.Tharushi Vinodya - Mr. Suneth Udayanga")
    st.write("Team Supervisor : Mr. Husni Mohamed ")
    
with st.container():
    st.write("---")
    st.markdown("<h6 style='text-align: center;'>Alpha Research Team - Faculty of Technology, Rajarata University of Sri Lanka (2024 - 2025)</h6>", unsafe_allow_html=True)
