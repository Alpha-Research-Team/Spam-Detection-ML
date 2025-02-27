import sdpmodel2.spamdetection as sd
import streamlit as st

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
                result = sd.predict(rt_message)
                if result['consensus'] == 'Spam':
                    st.markdown(
                        f"<div class='result-box spam'>❌ This message is <b>SPAM</b>.</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='result-box not-spam'>✅ This message is <b>NOT SPAM</b>.</div>",
                        unsafe_allow_html=True
                    )

with right_column:
    st.subheader("Our Models")
    left_column, right_column = st.columns(2)
    with left_column:
        nb_accuracy = (sd.model1Accuracy() * 100)
        st.progress(nb_accuracy/100)
        st.write(f"Naive Bayes (Multinomial) - Accuracy: {nb_accuracy:.2f}%")
        
        lr_accuracy = (sd.model2Accuracy() * 100)
        st.progress(lr_accuracy/100)
        st.write(f"Logistic Regression - Accuracy: {lr_accuracy:.2f}%")
       
    # Display with progress bars
    
    with right_column:
        gn_accuracy = (sd.model3Accuracy() * 100)
        st.progress(gn_accuracy/100)
        st.write(f"Naive Bayes (Gaussian) - Accuracy: {gn_accuracy:.2f}%")
        
        bn_accuracy = (sd.model4Accuracy() * 100)
        st.progress(bn_accuracy/100)
        st.write(f"Naive Bayes (Bernoulli) - Accuracy: {bn_accuracy:.2f}%")
        
    nmf_accuracy = (sd.newModelAccuracy() * 100)
    with st.container():
        st.progress(nmf_accuracy/100)
        st.write(f"New Model - Accuracy: {nmf_accuracy:.2f}%")
        
        
with st.container():
    st.write("---")
    st.write("Alpha Research Team RUSL (2024 - 2025)")
    st.write("Team : Kavishan - Rashmika - Tharushi - Suneth")
    
