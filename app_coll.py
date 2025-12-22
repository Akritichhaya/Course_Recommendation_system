
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Hybrid Course Recommendation System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Course Recommendation System")
st.markdown("""
**Hybrid Recommendation System Features:**

- Uses Domain-based and Collaborative Filtering
- 75% Domain-based recommendations (shown first)
- 15% Collaborative recommendations (shown last)
- Dynamic Cold-Start handling for new users
""")

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
DATA_PATH = "online_course_recommendation_v2.csv"
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------------
# DOMAIN EXTRACTION
# -------------------------------------------------------
def extract_domain(course_name):
    name = str(course_name).lower()

    if any(k in name for k in ["python", "machine learning", "ml", "ai", "data"]):
        return "Programming"
    if any(k in name for k in ["devops", "cloud", "aws", "azure", "ci/cd"]):
        return "DevOps"
    if any(k in name for k in ["network", "cyber", "security"]):
        return "Networking"
    if any(k in name for k in ["blockchain", "crypto"]):
        return "Blockchain"
    if any(k in name for k in ["finance", "trading", "stock"]):
        return "Finance"
    if any(k in name for k in ["marketing", "digital marketing"]):
        return "Marketing"
    if any(k in name for k in ["design", "canva", "graphic"]):
        return "Design"
    return "Other"

df["domain"] = df["course_name"].apply(extract_domain)

# -------------------------------------------------------
# USER → COURSE MAPPINGS
# -------------------------------------------------------

# Internal logic (course_id)
user_courses_id = (
    df.groupby("user_id")["course_id"]
    .apply(set)
    .to_dict()
)

# UI display (course_name)
user_courses_name = (
    df.groupby("user_id")["course_name"]
    .apply(list)
    .to_dict()
)

# -------------------------------------------------------
# COLLABORATIVE FILTERING
# -------------------------------------------------------
def get_similar_users(target_user_id):
    if target_user_id not in user_courses_id:
        return []

    target_courses = user_courses_id[target_user_id]

    return [
        uid for uid, courses in user_courses_id.items()
        if uid != target_user_id and len(courses & target_courses) > 0
    ]


def collaborative_recommend(user_id):
    similar_users = get_similar_users(user_id)

    if not similar_users:
        return []

    taken_courses = user_courses_id.get(user_id, set())

    return (
        df[df["user_id"].isin(similar_users)]
        .loc[~df["course_id"].isin(taken_courses)]
        ["course_name"]
        .value_counts()
        .index
        .tolist()
    )

# -------------------------------------------------------
# DOMAIN-BASED RECOMMENDATION
# -------------------------------------------------------
def domain_recommend(user_id, top_n):

    taken_courses = user_courses_id.get(user_id, set())
    user_domains = set(df[df["user_id"] == user_id]["domain"])

    domain_df = df[
        (df["domain"].isin(user_domains)) &
        (~df["course_id"].isin(taken_courses))
    ]

    ranked = (
        domain_df.groupby("course_name", as_index=False)
        .agg({
            "rating": "mean",
            "enrollment_numbers": "max"
        })
    )

    ranked["score"] = ranked["rating"] * np.log(ranked["enrollment_numbers"] + 1)

    return ranked.sort_values(by="score", ascending=False).head(top_n)

# -------------------------------------------------------
# POPULARITY-BASED MODEL (COLD START)
# -------------------------------------------------------
def popularity_recommend(top_n):

    pop_df = (
        df.groupby("course_name", as_index=False)
        .agg({
            "rating": "mean",
            "enrollment_numbers": "max"
        })
    )

    pop_df["score"] = pop_df["rating"] * np.log(pop_df["enrollment_numbers"] + 1)

    return pop_df.sort_values(by="score", ascending=False).head(top_n)

# -------------------------------------------------------
# HYBRID MODEL
# -------------------------------------------------------
def hybrid_recommend(user_id, top_n):

    # 🔴 Cold-start user
    if user_id not in user_courses_id:
        return popularity_recommend(top_n)

    # -------- Domain Based (75%) --------
    domain_k = int(top_n * 0.75)
    domain_part = domain_recommend(user_id, domain_k)

    # -------- Collaborative (15%) --------
    collab_courses = collaborative_recommend(user_id)

    collab_df = df[
        (df["course_name"].isin(collab_courses)) &
        (~df["course_name"].isin(domain_part["course_name"]))
    ]

    collab_part = (
        collab_df.groupby("course_name", as_index=False)
        .agg({
            "rating": "mean",
            "enrollment_numbers": "max"
        })
    )

    collab_part["score"] = (
        collab_part["rating"] * np.log(collab_part["enrollment_numbers"] + 1)
    )

    collab_part = collab_part.sort_values(
        by="score", ascending=False
    ).head(top_n - len(domain_part))

    return pd.concat([domain_part, collab_part], ignore_index=True)

# -------------------------------------------------------
# SIDEBAR UI
# -------------------------------------------------------

# -------------------------------------------------------
# USER DROPDOWN (Enrolled / Not Enrolled)
# -------------------------------------------------------

st.sidebar.title("🔍 User Selection")

# 1️⃣ Decide which user IDs you want to show
user_ids_to_show = list(range(1, 50000))  # 71–78 (you can change this)

# 2️⃣ Enrolled users from CSV
enrolled_users = set(df["user_id"].unique())

# 3️⃣ Build dropdown options
user_dropdown_options = []

for uid in user_ids_to_show:
    if uid in enrolled_users:
        user_dropdown_options.append(f"{uid} (Enrolled)")
    else:
        user_dropdown_options.append(f"{uid} (Not Enrolled)")

# 4️⃣ Dropdown
selected_label = st.sidebar.selectbox(
    "Select User ID",
    user_dropdown_options
)

selected_user_id = int(selected_label.split()[0])


top_n = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=6
)

# -------------------------------------------------------
# SIDEBAR: PREVIOUSLY ENROLLED COURSES
# -------------------------------------------------------
st.sidebar.markdown("### 📘 Previously Enrolled Courses")

if selected_user_id in user_courses_name:
    for course in user_courses_name[selected_user_id]:
        st.sidebar.markdown(f"• {course}")
else:
    st.sidebar.info("No enrolled courses found (New User)")

# -------------------------------------------------------
# MAIN OUTPUT
# -------------------------------------------------------
st.subheader("📌 Recommendation Results")

is_new_user = selected_user_id not in df["user_id"].values

if is_new_user:
    st.info("👤 New User Detected → Popularity-Based Recommendations")
else:
    st.success("✅ Enrolled User → Hybrid Recommendations")

if st.sidebar.button("Get Recommendations"):
    result = hybrid_recommend(selected_user_id, top_n)

    st.dataframe(
        result.rename(columns={
            "course_name": "Course Name",
            "rating": "Avg Rating",
            "enrollment_numbers": "Enrollments",
            "score": "Relevance Score"
        }),
        use_container_width=True
    )

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("---")
st.caption(
    # **Recommendation Strategy**
    
    "for all unseen users using a Popularity-based model."
)
