import sdpmodel2.spamdetection as sd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Spam Detection using ML",
    page_icon=":robot_face:",
    layout="centered"
)

# Custom CSS for modern design
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
            font-family: 'Inter', sans-serif;
        }
        .stButton>button {
            background: rgb(111,159,255);
background: linear-gradient(90deg, rgba(111,159,255,1) 0%, rgba(197,214,254,1) 25%, rgba(197,214,254,1) 75%, rgba(111,159,255,1) 100%);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
            color: #ffffff;
            padding: 0.8em 2.5em;
            border-radius: 12px;
            border: none;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 40%;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .result-box {
            padding: 1.5em;
            border-radius: 16px;
            font-size: 18px;
            text-align: center;
            margin-top: 2em;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .spam {
            background-color: #fff0f0;
            color: #ff4d4d;
            box-shadow: 0 4px 12px rgba(255, 77, 77, 0.2);
        }
        .not-spam {
            background-color: #f0fff4;
            color: #00cc66;
            box-shadow: 0 4px 12px rgba(0, 204, 102, 0.2);
        }
        hr {
            border: none;
            border-top: 1px solid #e0e0e0;
            margin: 2em 0;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #666;
        }
        h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 0.5em;
        }
        p {
            color: #555;
            font-size: 1.1em;
            line-height: 1.6;
        }
        .input-box {
            margin-bottom: 1.5em;
        }
        .input-box input {
            width: 100%;
            padding: 0.8em;
            border-radius: 12px;
            border: 1px solid #ddd;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .input-box input:focus {
            border-color: #6a11cb;
            box-shadow: 0 0 8px rgba(106, 17, 203, 0.2);
            outline: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)



# App header
st.markdown(
    """
    <h1 style='text-align: center;'>📧 Spam Detection System</h1>
    <p style='text-align: center;'>Instantly check if a message is spam using Machine Learning.</p>
    <p style='text-align: center; font-weight: bold; color: #6a11cb;'>Algorithm Used: Naive Bayes (Multinomial) & Linear Model (LogisticRegression)</p>
    """,
    unsafe_allow_html=True
)

# Input field
st.markdown("<div class='input-box'>", unsafe_allow_html=True)
rt_message = st.text_input(
    'Enter your message:',
    placeholder='Type your message here...'
)


# Validation button
if st.button('Validate Message 🚀'):
    if rt_message.strip() == '':
        st.warning('⚠️ Please enter a valid message.')
    else:
        result = sd.predict(rt_message)
        if result == 'Spam':
            st.markdown(
                f"<div class='result-box spam'>❌ This message is <b>SPAM</b>.</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box not-spam'>✅ This message is <b>NOT SPAM</b>.</div>",
                unsafe_allow_html=True
            )

# Footer
st.markdown(
    """
    <hr>
    <p class='footer'><strong>Team Members:</strong> Kavishan, Rashmika, Tharushi, Suneth</p>
    <p class='footer'>Alpha Research Team - RUSL</p>
    """,
    unsafe_allow_html=True
)

# Close card layout
st.markdown("</div>", unsafe_allow_html=True)