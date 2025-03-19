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
    left_column, middle_column, right_column = st.columns((3,1,2))
    with left_column:
        st.subheader("Enter your message:")
        rt_message = st.text_input("Text Only", placeholder="Type your message here...")
        if st.button('Validate Message 🚀'):
            if rt_message.strip() == '':
                st.warning('⚠️ Please enter a valid message.')
            else:
                result = spam_detection.stack_predict(rt_message)
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
                st.write('---')
                # if result['multinomial_nb']:
                #     st.write('- Multinomial :', result['multinomial_nb'])
                # if result['logistic_regression']:
                #     st.write('- Logistic Regression :', result['logistic_regression'])
                # if result['bernoulli_nb']:
                #     st.write('- Bernoulli :', result['bernoulli_nb'])
                # if result['random_forest']:
                #     st.write('- Random Forest :', result['random_forest'])
                    
                    
                # st.write('- Multinomial :', result['multinomial_nb'])
                # st.write('- Logistic Regression :', result['logistic_regression'])
                # st.write('- Bernoulli :', result['bernoulli_nb'])
                # st.write('- Random Forest :', result['random_forest'])
    with right_column:
        st.subheader("Our Models & Accuracy")
        left_column, right_column = st.columns(2)
        with left_column:
            nb_accuracy = (spam_detection.model1_accuracy() * 100)
            st.progress(nb_accuracy/100)
            st.write(f"Naive Bayes (Multinomial) : {nb_accuracy:.2f}%")
            
            lr_accuracy = (spam_detection.model2_accuracy() * 100)
            st.progress(lr_accuracy/100)
            st.write(f"Logistic Regression : {lr_accuracy:.2f}%")
       
    # Display with progress bars
    
        with right_column:
            bn_accuracy = (spam_detection.model3_accuracy() * 100)
            st.progress(bn_accuracy/100)
            st.write(f"Naive Bayes (Bernoulli) : {bn_accuracy:.2f}%")
            
            rf_accuracy = (spam_detection.model4_accuracy() * 100)
            st.progress(rf_accuracy/100)
            st.write(f"Random Forest : {rf_accuracy:.2f}%")
            
            # bn_accuracy = (sd.model4Accuracy() * 100)
            # st.progress(bn_accuracy/100)
            # st.write(f"Naive Bayes (Bernoulli) - Accuracy: {bn_accuracy:.2f}%")
            
        nmf_accuracy = (spam_detection.stack_model_accuracy() * 100)
        # dataset_size = spam_detection.dataset_info()
        with st.container():
            st.progress(nmf_accuracy/100)
            st.write(f"Stack Model - Accuracy: {nmf_accuracy:.2f}%")
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
    st.write("Alpha Research Team RUSL (2024 - 2025)")
    st.write("Team : Kavishan - Rashmika - Tharushi - Suneth")
