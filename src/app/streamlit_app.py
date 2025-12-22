import streamlit as st
import pandas as pd
import os
import sys
# ---- FIX PYTHON PATH FOR STREAMLIT ----
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.services.recommender import RestaurantRecommender

from src.services.recommender import RestaurantRecommender

from src.services.recommender import RestaurantRecommender

# ---------------- Page Config ---------------- #
st.set_page_config(
    page_title="Zomato Restaurant Recommender",
    page_icon="🍽️",
    layout="wide"
)

# ---------------- Custom CSS ---------------- #
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1504674900247-0877df9cc836");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .main {
        background-color: rgba(0, 0, 0, 0.75);
        padding: 2rem;
        border-radius: 12px;
    }

    h1, h2, h3, h4, h5 {
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }

    p, label {
        color: #f0f0f0;
        font-size: 16px;
    }

    .card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- App Title ---------------- #
st.markdown(
    "<div class='main'><h1>🍽️ Zomato Restaurant Recommendation System</h1>"
    "<p>AI-powered restaurant discovery using Deep Learning & NLP</p></div>",
    unsafe_allow_html=True
)

# ---------------- Sidebar ---------------- #
st.sidebar.header("🔍 Recommendation Filters")

cuisine = st.sidebar.text_input(
    "Enter Cuisine",
    placeholder="e.g., Biryani, Chinese, Italian"
)

top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=3,
    max_value=15,
    value=5
)

# ---------------- Recommender ---------------- #
@st.cache_resource
def load_recommender():
    return RestaurantRecommender()

recommender = load_recommender()

# ---------------- Action ---------------- #
if st.sidebar.button("🍴 Recommend Restaurants"):
    if not cuisine.strip():
        st.warning("Please enter a cuisine name.")
    else:
        with st.spinner("Finding the best restaurants for you..."):
            results = recommender.recommend(cuisine, top_n)

        if not results:
            st.error("No restaurants found for this cuisine.")
        else:
            st.markdown("<div class='main'><h2>✨ Top Recommendations</h2></div>", unsafe_allow_html=True)

            for idx, row in enumerate(results, start=1):
                st.markdown(
                    f"""
                    <div class="card">
                        <h3>#{idx} {row['Name']}</h3>
                        <p><b>Cuisines:</b> {row['Cuisines']}</p>
                        <p><b>Cost for Two:</b> ₹{row['Cost']}</p>
                        <p><b>Predicted Rating:</b> ⭐ {row['predicted_score']:.2f}</p>
                        <p><b>Recommendation Score:</b> 🔥 {row['final_score']:.2f}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ---------------- Footer ---------------- #
st.markdown(
    """
    <hr style="border:1px solid #444;">
    <p style="text-align:center; color:#bbb;">
    Built with ❤️ using Deep Learning, Word2Vec, FastAPI & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
