import streamlit as st
import pandas as pd
import numpy as np

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_data
def load_data():
    DATA_PATH = r"online_course_recommendation_v2.csv"
    return pd.read_csv(DATA_PATH)

df = load_data()

# ======================================================
# DOMAIN EXTRACTION (STRONG, DATA-AWARE)
# ======================================================
def extract_domain(course_name):
    name = str(course_name).lower()

    if any(k in name for k in ["python", "machine learning", "ml", "ai", "data"]):
        return "Programming"

    if any(k in name for k in ["devops", "deployment", "ci/cd", "cloud", "aws", "azure"]):
        return "DevOps"

    if any(k in name for k in ["network", "system", "cyber", "security"]):
        return "Networking"

    if any(k in name for k in ["blockchain", "crypto", "decentralized"]):
        return "Blockchain"

    if any(k in name for k in ["finance", "trading", "stock"]):
        return "Finance"

    if any(k in name for k in ["marketing"]):
        return "Marketing"

    if any(k in name for k in ["design", "canva", "graphic"]):
        return "Design"

    if any(k in name for k in ["fitness", "nutrition"]):
        return "Health"

    return "Other"

df["domain"] = df["course_name"].apply(extract_domain)

# ======================================================
# USER → COURSE MAPPING
# ======================================================
df_cf = df[["user_id", "course_id", "course_name"]].drop_duplicates()

@st.cache_data
def build_user_courses(data):
    return data.groupby("user_id")["course_id"].apply(set).to_dict()

user_courses = build_user_courses(df_cf)

# ======================================================
# COLLABORATIVE FILTERING (SUPPORTING ROLE)
# ======================================================
def get_similar_users(user_id):
    if user_id not in user_courses:
        return []

    target_courses = user_courses[user_id]
    return [
        uid for uid, courses in user_courses.items()
        if uid != user_id and len(courses & target_courses) > 0
    ]

def collaborative_recommend(user_id, n=10):
    if user_id not in user_courses or len(user_courses[user_id]) == 0:
        return []

    similar_users = get_similar_users(user_id)

    candidates = (
        df_cf[df_cf["user_id"].isin(similar_users)]
        .loc[~df_cf["course_id"].isin(user_courses[user_id])]
    )

    return candidates["course_name"].value_counts().head(n).index.tolist()

# # ======================================================
# # POPULARITY MODEL (COLD START)
# # ======================================================
# def popularity_recommend(top_n):
#     temp = (
#         df.groupby("course_name", as_index=False)
#         .agg({"rating": "mean", "enrollment_numbers": "max"})
#     )

#     temp["score"] = temp["rating"] * np.log(temp["enrollment_numbers"] + 1)

#     return (
#         temp.sort_values(by="score", ascending=False)
#         .head(top_n)
#         .reset_index(drop=True)
#     )

# ======================================================
# 🔥 TWO-STAGE HYBRID RECOMMENDER (85% DOMAIN + 15% CF)
# ======================================================
def hybrid_recommend(user_id, top_n):
    # -------- Cold Start --------
    if user_id not in user_courses or len(user_courses[user_id]) == 0:
        return popularity_recommend(top_n)

    # -------- User Domains --------
    user_domains = set(df[df["user_id"] == user_id]["domain"])

    # =============================
    # STAGE 1: DOMAIN-BASED (85%)
    # =============================
    domain_df = df[
        (df["domain"].isin(user_domains)) &
        (~df["course_name"].isin(
            df[df["user_id"] == user_id]["course_name"]
        ))
    ]

    domain_ranked = (
        domain_df.groupby("course_name", as_index=False)
        .agg({
            "rating": "mean",
            "enrollment_numbers": "max"
        })
    )

    domain_ranked["score"] = (
        domain_ranked["rating"] *
        np.log(domain_ranked["enrollment_numbers"] + 1)
    )

    domain_ranked = domain_ranked.sort_values(
        by="score", ascending=False
    )

    domain_k = max(1, int(top_n * 0.85))
    domain_results = domain_ranked.head(domain_k)

    # =============================
    # STAGE 2: COLLABORATIVE (15%)
    # =============================
    collab_courses = collaborative_recommend(user_id, n=top_n * 2)

    collab_df = df[
        (df["course_name"].isin(collab_courses)) &
        (~df["course_name"].isin(domain_results["course_name"])) &
        (~df["course_name"].isin(
            df[df["user_id"] == user_id]["course_name"]
        ))
    ]

    collab_ranked = (
        collab_df.groupby("course_name", as_index=False)
        .agg({
            "rating": "mean",
            "enrollment_numbers": "max"
        })
    )

    collab_ranked["score"] = (
        collab_ranked["rating"] *
        np.log(collab_ranked["enrollment_numbers"] + 1)
    )

    collab_ranked = collab_ranked.sort_values(
        by="score", ascending=False
    )

    collab_k = top_n - len(domain_results)
    collab_results = collab_ranked.head(collab_k)

    # =============================
    # FINAL MERGE (ORDER PRESERVED)
    # =============================
    final_df = pd.concat(
        [domain_results, collab_results],
        ignore_index=True
    )

    return final_df[["course_name", "rating", "enrollment_numbers"]]

# ======================================================
# SIDEBAR UI
# ======================================================
st.sidebar.title("🎓 Course Recommender")

user_id = st.sidebar.selectbox(
    "Select User ID",
    sorted(df_cf["user_id"].unique())
)

st.sidebar.subheader("📘 Previously Enrolled Courses")
past_courses = (
    df[df["user_id"] == user_id]["course_name"]
    .drop_duplicates()
    .tolist()
)

if past_courses:
    for c in past_courses:
        st.sidebar.markdown(f"• {c}")
else:
    st.sidebar.info("Cold-start user")

top_n = st.sidebar.slider("Number of Recommendations", 1, 10)
btn = st.sidebar.button("Get Recommendations")

# ======================================================
# MAIN OUTPUT
# ======================================================
st.title("📚 Recommended Courses")

st.markdown("""
**Recommendation Strategy**
- **85% Domain-based recommendations (shown first)**
- **15% Collaborative recommendations (shown last)**

""")

if btn:
    recs = hybrid_recommend(user_id, top_n)
    st.dataframe(recs, use_container_width=True)

st.markdown("---")
st.caption(
    "Two-stage hybrid recommendation system: "
    "domain-first ranking with collaborative tail support."
)
