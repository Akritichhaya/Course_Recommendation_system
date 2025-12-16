import streamlit as st
import pandas as pd
import numpy as np

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Online Course Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# ==============================
# Header Section (ONLY ONCE)
# ==============================
st.title("🎓 Online Course Recommendation System")

st.markdown("""
✅ **Model Used:** Collaborative Filtering using SVD  
📊 **Input:** User ID + Number of Recommendations  

""")

st.markdown("---")

# ==============================
# Load Dataset
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("online_course_recommendation_v2.csv")
    df = df.drop_duplicates().dropna()
    return df

df = load_data()

# ==============================
# Train SVD Model
# ==============================
from surprise import Dataset, Reader, SVD

@st.cache_resource
def train_model(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        df[['user_id', 'course_id', 'rating']],
        reader
    )
    trainset = data.build_full_trainset()

    model = SVD(
        n_factors=20,   # RAM safe
        n_epochs=20,
        random_state=42
    )
    model.fit(trainset)
    return model

svd_model = train_model(df)

# ==============================
# Recommendation Function
# ==============================
def recommend_courses(user_id, n_recommendations):
    all_courses = df['course_id'].unique()
    taken_courses = df[df['user_id'] == user_id]['course_id'].unique()

    unseen_courses = [c for c in all_courses if c not in taken_courses]

    predictions = []
    for course_id in unseen_courses:
        pred = svd_model.predict(user_id, course_id)
        predictions.append((course_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_course_ids = [c[0] for c in predictions[:n_recommendations]]

    result = (
        df[df['course_id'].isin(top_course_ids)]
        .drop_duplicates(subset='course_id')
        .head(n_recommendations)
        [['course_id', 'course_name',
          'difficulty_level', 'course_price']]
        .reset_index(drop=True)
    )

    return result

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("🔧 Recommendation Settings")

user_id = st.sidebar.selectbox(
    "Select User ID",
    sorted(df['user_id'].unique())
)

n_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

# ==============================
# Generate Recommendations
# ==============================
if st.sidebar.button("🎯 Get Recommendations"):
    recommendations = recommend_courses(user_id, n_recommendations)

    st.markdown(
        f"<h3 style='text-align:center;'>🎯 Showing Top {n_recommendations} Recommended Courses</h3>",
        unsafe_allow_html=True
    )

    st.dataframe(recommendations, use_container_width=True)
