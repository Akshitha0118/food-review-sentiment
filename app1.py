import streamlit as st
import pickle
import re
import os
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')

from nltk.stem.porter import PorterStemmer


st.set_page_config(
    page_title="Food Review Sentiment Analyzer",
    layout="wide"
)


# Custom CSS

st.markdown("""
<style>
.main {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 12px;
}
.title {
    text-align: center;
    color: #ff4b4b;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
}
.result-positive {
    color: green;
    font-size: 26px;
    font-weight: bold;
}
.result-negative {
    color: red;
    font-size: 26px;
    font-weight: bold;
}
.warning {
    color: orange;
    font-size: 20px;
    font-weight: bold;
}
.food-box {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Models
# -----------------------------
MODEL_DIR = 'models'

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(MODEL_DIR, "random_forest_model.pkl"), "rb") as f:
    model = pickle.load(f)

# -----------------------------
# NLP Preprocessing
# -----------------------------
ps = PorterStemmer()

def preprocess(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# -----------------------------
# Food Vocabulary
# -----------------------------
FOOD_ITEMS = [
    "pizza", "burger", "biryani", "rice", "chicken", "mutton",
    "paneer", "naan", "roti", "fries", "noodles", "pasta",
    "sandwich", "meal", "food", "dish", "restaurant",
    "taste", "service", "delivery", "menu", "samosa",
    "dosa", "idli", "vada", "curry", "salad"
]

def extract_food_items(text):
    text = text.lower()
    found = [item for item in FOOD_ITEMS if item in text]
    return list(set(found))  # remove duplicates

# -----------------------------
# UI Layout
# -----------------------------
st.markdown('<div class="title">🍽️ Food Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect food items & analyze review sentiment</div>', unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns([2, 1])

with col1:
    review = st.text_area("📝 Enter your food review:", height=180)
    analyze = st.button("🔍 Analyze Review")

with col2:
    st.markdown("### 🍔 Detected Food Items")
    food_container = st.empty()

# -----------------------------
# Prediction Logic
# -----------------------------
if analyze:
    if review.strip() == "":
        st.warning("Please enter a review.")
    
    else:
        detected_foods = extract_food_items(review)

        if not detected_foods:
            food_container.markdown(
                '<div class="food-box">❌ No food items detected</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="warning">⚠️ This model works only for food-related reviews.</div>',
                unsafe_allow_html=True
            )
        else:
            # Show food items
            food_container.markdown(
                "<div class='food-box'>" +
                "<br>".join([f"• {food}" for food in detected_foods]) +
                "</div>",
                unsafe_allow_html=True
            )

            # Predict sentiment
            cleaned_review = preprocess(review)
            X_new = tfidf.transform([cleaned_review])
            prediction = model.predict(X_new)[0]

            if prediction == 1:
                st.markdown(
                    '<div class="result-positive">✅ Positive Review 😊</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-negative">❌ Negative Review 😞</div>',
                    unsafe_allow_html=True
                )

st.write("")
st.caption("⚡ TF-IDF + Random Forest | NLP Food Review System")


