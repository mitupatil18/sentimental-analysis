import pandas as pd 
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Load model and vectorizer
model = pk.load(open('model.pkl','rb'))
scaler = pk.load(open('scaler.pkl','rb'))

# Set page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

# Add a title and description
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("""
    Welcome to the Movie Review Sentiment Analyzer! 
    Enter a movie review below, and we'll predict whether it's positive or negative.
""")

# Create a sidebar with example reviews
st.sidebar.header("Example Reviews")
st.sidebar.write("Here are some example reviews you can try:")
examples = [
    "The movie was fantastic! I loved every part of it.",
    "It was a waste of time. The plot was boring.",
    "An absolute masterpiece with brilliant performances.",
    "Not as good as I expected. It was just okay."
]
for example in examples:
    st.sidebar.write(f"- {example}")

# Add a text input for the review
review = st.text_area('Enter Movie Review', height=200)

# Add a button for prediction
if st.button('Predict'):
    if review.strip() != "":  # Check if the input is not empty
        review_transformed = scaler.transform([review])
        result = model.predict(review_transformed)
        if result[0] == 0:
            st.error('Negative Review 😞')
        else:
            st.success('Positive Review 😊')
    else:
        st.warning('Please enter a valid review')

# Add a footer with some style
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
        }
    </style>
    <div class="footer">
    Developed with ❤️ by <a href="https://github.com/mitupatil18">Mitali Patil</a>
</div>

""", unsafe_allow_html=True)
