import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Background
def set_bg():

    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://i0.wp.com/news.shreestar.com/wp-content/uploads/2023/08/desktop-wallpaper-flipkart-app-install-campaign-flipkart.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()



# Download once
nltk.download('stopwords')
nltk.download('wordnet')


# Load Model & Vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Text Cleaning
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# App UI
st.set_page_config(page_title="Flipkart Review Sentiment", page_icon="⭐")

st.title("🛒 Flipkart Review Sentiment Analysis")

st.write("Enter your product review below to check sentiment.")


review = st.text_area("✍️ Enter Review Here")


if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first!")

    else:

        clean_review = clean_text(review)

        data = vectorizer.transform([clean_review])

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.success("✅ Sentiment: Positive 😊")

        else:
            st.error("❌ Sentiment: Negative 😠")


st.markdown("---")
st.markdown("### 📌 Project: Sentiment Analysis of Flipkart Reviews")
st.markdown("Developed by Krishna")
