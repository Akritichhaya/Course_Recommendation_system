import streamlit as st
import pandas as pd
import random
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# -----------------------------------------------------------
# CUSTOM CSS FOR BEAUTIFUL UI
# -----------------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #f4f6fa;
}

/* Card Design */
.card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-left: 6px solid #4c83ff;
}

.card:hover {
    transform: scale(1.01);
    transition: 0.2s ease-in-out;
    box-shadow: 0 8px 20px rgba(0,0,0,0.18);
}

/* Title */
.top-header {
    font-size: 45px;
    font-weight: 800;
    color: #1b263b;
    margin-bottom: -10px;
}

.course-title {
    font-size: 20px;
    font-weight: 700;
    color: #1b263b;
}

.instructor {
    font-size: 15px;
    font-weight: 600;
    color: #4c83ff;
}

.difficulty {
    font-size: 14px;
    color: #696969;
}

.star-rating {
    font-size: 18px;
    color: #f4c430;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("online_course_recommendation_v2.csv")

    df['certification_offered'] = df['certification_offered'].map({'Yes':1, 'No':0})
    df['study_material_available'] = df['study_material_available'].map({'Yes':1, 'No':0})
    df['difficulty_level_encoded'] = df['difficulty_level'].map({'Beginner':1, 'Intermediate':2, 'Advanced':3})

    return df

df = load_data()

# -----------------------------------------------------------
# TITLE
# -----------------------------------------------------------
st.markdown("<h1 class='top-header'>🎓 Course Recommendation System</h1>", unsafe_allow_html=True)
st.write("AI-powered recommendations using **Surprise SVD Collaborative Filtering**.")
st.write("---")

# -----------------------------------------------------------
# SIDEBAR SETTINGS
# -----------------------------------------------------------
st.sidebar.title("🔧 Settings")

user_list = sorted(df['user_id'].unique())
selected_user = st.sidebar.selectbox("Select User ID:", user_list)

num_rec = st.sidebar.slider("Number of Recommendations:", 1, 10, 5)

# -----------------------------------------------------------
# BUILDING SVD MODEL
# -----------------------------------------------------------
reader = Reader(rating_scale=(0, 1))  
data = Dataset.load_from_df(df[['user_id', 'course_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

st.sidebar.success("✅ SVD Model Loaded Successfully!")

# -----------------------------------------------------------
# RECOMMENDATION FUNCTION (SVD BASED)
# -----------------------------------------------------------
def recommend_courses(user_id, n=5):
    all_courses = df['course_id'].unique()
    rated = df[df['user_id'] == user_id]['course_id'].unique()

    # Courses user has NOT seen
    remaining = [c for c in all_courses if c not in rated]

    # Predict ratings
    predictions = [(c, model.predict(user_id, c).est) for c in remaining]

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_ids = [cid for cid, _ in predictions[:n]]

    return df[df['course_id'].isin(top_ids)][[
        'course_id', 'course_name', 'instructor', 'difficulty_level', 'rating'
    ]].drop_duplicates()

# -----------------------------------------------------------
# USER DASHBOARD
# -----------------------------------------------------------
st.subheader("👤 User Profile Overview")

user_data = df[df['user_id'] == selected_user]

total_courses = len(user_data)
avg_rating = round(user_data['rating'].mean() * 5, 2)
total_time = round(user_data['time_spent_hours'].sum(), 2)
difficulty_pref = user_data['difficulty_level'].mode()[0]
fav_instructor = user_data['instructor'].mode()[0]

st.markdown(
    f"""
    <div class="card">
        <h3>📌 User Summary</h3>
        <p><b>User ID:</b> {selected_user}</p>
        <p><b>Total Courses Taken:</b> {total_courses}</p>
        <p><b>Average Rating Given:</b> ⭐ {avg_rating}</p>
        <p><b>Total Time Spent:</b> ⏳ {total_time} hrs</p>
        <p><b>Preferred Difficulty:</b> {difficulty_pref}</p>
        <p><b>Favorite Instructor:</b> 👨‍🏫 {fav_instructor}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------
# SEARCH COURSES
# -----------------------------------------------------------
st.subheader("🔎 Search Courses")

search_query = st.text_input("Search by Course Name:", "")

if search_query.strip():
    results = df[df['course_name'].str.contains(search_query, case=False)]
    st.write(f"Found {len(results)} courses:")

    for _, row in results.iterrows():
        stars = "⭐" * int(round(row['rating'] * 5))
        st.markdown(
            f"""
            <div class="card">
                <h4 class="course-title">{row['course_name']}</h4>
                <p class="instructor">👨‍🏫 {row['instructor']}</p>
                <p class="difficulty">📘 Difficulty: {row['difficulty_level']}</p>
                <p class="star-rating">{stars}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.write("---")

# -----------------------------------------------------------
# SHOW RECOMMENDATIONS
# -----------------------------------------------------------
if st.sidebar.button("Get Recommendations"):
    st.subheader(f"🎯 Top {num_rec} Courses For You")

    recs = recommend_courses(selected_user, num_rec)

    for _, row in recs.iterrows():
        img_url = f"https://source.unsplash.com/600x300/?education,learning,{random.randint(1,100)}"
        stars = "⭐" * int(round(row['rating'] * 5))

        st.markdown(
            f"""
            <div class="card">
                <img src="{img_url}" width="100%" style="border-radius:10px;">
                <h4 class="course-title">{row['course_name']}</h4>
                <p class="instructor">👨‍🏫 {row['instructor']}</p>
                <p class="difficulty">📘 Difficulty: {row['difficulty_level']}</p>
                <p class="star-rating">{stars}</p>
                <p><b>Course ID:</b> {row['course_id']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------------------------------------
# SIMILAR COURSES (USING SVD LATENT FACTORS)
# -----------------------------------------------------------
st.subheader("📚 Find Similar Courses")

df['cid_cat'] = df['course_id'].astype('category')
course_index = {cid: i for i, cid in enumerate(df['cid_cat'].cat.categories)}

def similar_courses(course_id, n=5):
    item_factors = model.qi  
    idx = course_index[course_id]

    sims = cosine_similarity([item_factors[idx]], item_factors)[0]

    top_idx = sims.argsort()[-n-1:-1][::-1]
    similar_ids = [df['cid_cat'].cat.categories[i] for i in top_idx]

    return df[df['course_id'].isin(similar_ids)][['course_id','course_name','rating']].drop_duplicates()

selected_course = st.selectbox("Choose Course ID:", sorted(df['course_id'].unique()))

if st.button("Show Similar Courses"):
    sims = similar_courses(selected_course)

    for _, row in sims.iterrows():
        stars = "⭐" * int(round(row['rating'] * 5))
        st.markdown(
            f"""
            <div class="card">
                <h4 class="course-title">{row['course_name']}</h4>
                <p class="star-rating">{stars}</p>
                <p><b>Course ID:</b> {row['course_id']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
