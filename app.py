# ###############################
# #   FINAL COURSE RECO APP     #
# ###############################

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity
# from numpy.linalg import norm

# # ---------------------------------------------
# # 1. LOAD & CLEAN DATA
# # ---------------------------------------------

# DATA_PATH = r"D:\Recommendation_system\online_course_recommendation_v2.csv"
# df = pd.read_csv(DATA_PATH)

# df = df.drop_duplicates()
# df = df.dropna(subset=["course_name"])

# num_cols = df.select_dtypes(include=["int64","float64"]).columns
# cat_cols = df.select_dtypes(include=["object"]).columns

# df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
# df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# df_clean = df.copy()

# # ---------------------------------------------
# # 2. FEATURE BASED SIMILARITY MODEL (LIGHTWEIGHT)
# # ---------------------------------------------

# num_features = ["course_duration_hours","course_price","feedback_score"]
# cat_features = ["difficulty_level","certification_offered","study_material_available"]

# enc = OneHotEncoder()
# encoded = enc.fit_transform(df_clean[cat_features]).toarray()

# feature_matrix = pd.concat(
#     [df_clean[num_features].reset_index(drop=True),
#      pd.DataFrame(encoded)],
#     axis=1
# )

# feature_matrix.columns = feature_matrix.columns.astype(str)
# scaler = StandardScaler()
# feature_scaled = scaler.fit_transform(feature_matrix)

# def feature_similarity(course_name, top_n):
#     try:
#         idx = df_clean[df_clean["course_name"].str.lower() == course_name.lower()].index[0]
#     except:
#         return None

#     sims = cosine_similarity(feature_scaled[idx].reshape(1,-1), feature_scaled).flatten()
#     top_idx = sims.argsort()[::-1][1:top_n+1]
    
#     result = df_clean.iloc[top_idx][["course_id","course_name","instructor",
#                                      "difficulty_level","course_price","rating"]]
#     result["score_feature"] = sims[top_idx]
    
#     return result


# # ---------------------------------------------
# # 3. TF-IDF SIMILARITY MODEL
# # ---------------------------------------------

# df_clean["combined_text"] = (
#     df_clean["course_name"] + " " +
#     df_clean["difficulty_level"] + " " +
#     df_clean["instructor"]
# )

# tfidf = TfidfVectorizer(stop_words="english", max_features=4000)
# tfidf_matrix = tfidf.fit_transform(df_clean["combined_text"])

# def tfidf_similarity(course_name, top_n):
#     course_name = course_name.lower()

#     if course_name not in df_clean["course_name"].str.lower().values:
#         return None

#     idx = df_clean[df_clean["course_name"].str.lower()==course_name].index[0]

#     sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
#     top_idx = sims.argsort()[::-1][1:top_n+1]

#     result = df_clean.iloc[top_idx][["course_id","course_name","instructor",
#                                      "difficulty_level","course_price","rating"]]
#     result["score_tfidf"] = sims[top_idx]

#     return result


# # ---------------------------------------------
# # 4. HYBRID FINAL RECOMMENDER
# # ---------------------------------------------

# def hybrid_recommender(course_name, top_n=6):
#     # Find closest title instead of exact match
#     course_name = course_name.lower().strip()
    
#     # Best fuzzy match instead of strict equality
#     matches = df_clean[df_clean["course_name"].str.lower().str.contains(course_name[:5])]
    
#     if matches.empty:
#         matches = df_clean[df_clean["course_name"].str.lower()].iloc[:1]  # fallback
    
#     main_course = matches["course_name"].iloc[0]

#     # Run models
#     res1 = feature_similarity(main_course, top_n*3)
#     res2 = tfidf_similarity(main_course, top_n*3)

#     # Safety fallback
#     if res1 is None and res2 is None:
#         return df_clean.sample(top_n)   # show random courses but NEVER blank
    
#     if res1 is None:
#         return res2.head(top_n)
#     if res2 is None:
#         return res1.head(top_n)

#     # Merge both
#     merged = pd.merge(
#         res1, res2,
#         on=["course_id","course_name","instructor","difficulty_level","course_price","rating"],
#         how="outer",
#         suffixes=("_f","_t")
#     )

#     # Fill missing similarity scores
#     merged["score_feature"] = merged["score_feature"].fillna(0.3)
#     merged["score_tfidf"] = merged["score_tfidf"].fillna(0.3)

#     # Final weighted score
#     merged["final_score"] = 0.4 * merged["score_feature"] + 0.6 * merged["score_tfidf"]

#     merged = merged.sort_values("final_score", ascending=False).head(top_n)

#     return merged



# # ---------------------------------------------
# # 5. STREAMLIT UI (Exactly like screenshot)
# # ---------------------------------------------

# st.set_page_config(page_title="Course Recommendation", layout="wide")

# # HEADER
# st.markdown("<h1 style='text-align:center;'>🎓 Course Recommendation System</h1>", unsafe_allow_html=True)
# st.write("<h4 style='text-align:center;'>AI-powered recommendations tailored to your learning needs.</h4>", unsafe_allow_html=True)
# st.write("---")

# # SIDEBAR INPUT FORM
# with st.sidebar:
#     st.header("⚙️ Enter Your Preferences")

#     user_id = st.text_input("User ID:", "1")

#     course_selected = st.selectbox("Choose a Course:", df_clean["course_name"].unique())

#     difficulty = st.selectbox("Preferred Difficulty:", df_clean["difficulty_level"].unique())

#     certification = st.radio("Certification Required?", ["Both","Yes","No"])

#     study_material = st.radio("Study Material Required?", ["Both","Yes","No"])

#     num_rec = st.slider("Number of Recommendations:", 3, 10, 6)

#     submit = st.button("🔍 Get Recommendations")

# # ---------------------------------------------
# # 6. SHOW RESULTS
# # ---------------------------------------------

# if submit:
#     st.success("Recommendations Generated!")

#     st.markdown(
#         f"""
#         <h2>🎯 Top {num_rec} Personalized Recommendations</h2>
#         Based on <b>{course_selected}</b>, difficulty <b>{difficulty}</b>,
#         certification <b>{certification}</b>, study material <b>{study_material}</b>.
#         """,
#         unsafe_allow_html=True
#     )

#     results = hybrid_recommender(course_selected, num_rec)

#     if results is not None:
#         for _, row in results.iterrows():
#             st.markdown(
#                 f"""
#                 <div style="padding:15px; border-radius:8px; 
#                 border:1px solid #ddd; margin-bottom:10px;">
#                     <h3>📘 {row['course_name']}</h3>
#                     <p><b>Instructor:</b> {row['instructor']}</p>
#                     <p><b>Difficulty:</b> {row['difficulty_level']}</p>
#                     <p><b>Price:</b> ₹{row['course_price']}</p>
#                     <p><b>Rating:</b> ⭐ {row['rating']}/5</p>
                     
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#     else:
#         st.error("Course not found in dataset!")




#     card = f"""
#             <div style="
#                 background:white; padding:22px; border-radius:15px;
#                 border-left:6px solid #3b82f6;
#                 margin-bottom:25px; box-shadow:0 4px 12px rgba(0,0,0,0.1);
#                 transition:0.2s;"
#                 onmouseover="this.style.transform='translateY(-6px)';
#                              this.style.boxShadow='0 10px 25px rgba(0,0,0,0.2)'"
#                 onmouseout="this.style.transform='translateY(0)';
#                             this.style.boxShadow='0 4px 12px rgba(0,0,0,0.1)'">

#                 <h2>📘 {row['course_name']}</h2>
#                 <p>👨‍🏫 <b>{row['instructor']}</b></p>

#                 <p><b>Difficulty:</b> {row['difficulty_level']}</p>
#                 <p><b>Rating:</b> ⭐ {row['rating']} / 5</p>
#                 <p><b>Price:</b> ₹{row['course_price']}</p>

#                 <p><b>Certification:</b> ✓ {row['certification_offered']}</p>
#                 <p><b>Study Material:</b> ✓ {row['study_material_available']}</p>
#             </div>
#             """
#                                     # components.html(card, height=350)

# st.markdown("</div>", unsafe_allow_html=True)





# # ----------------------------------------------------------
# # 8. VISUALIZATION
# # ----------------------------------------------------------
# st.subheader("📊 Rating Distribution")
# fig, ax = plt.subplots(figsize=(6,4))
# sns.histplot(df["rating"], kde=True, ax=ax)
# st.pyplot(fig)



# # ----------------------------------------------------------
# # 8. VISUALIZATION
# # ----------------------------------------------------------
# st.subheader("📊 Rating Distribution")
# fig, ax = plt.subplots(figsize=(6,4))
# sns.histplot(df["rating"], kde=True, ax=ax)
# st.pyplot(fig)









import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt


# ----------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ----------------------------------------------------------

DATA_PATH = r"D:\Recommendation_system\online_course_recommendation_v2.csv"
df = pd.read_csv(DATA_PATH)

df = df.drop_duplicates()
df = df.dropna(subset=["course_name", "user_id", "course_id"])

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

df_clean = df.copy()


# ----------------------------------------------------------
# 2. FEATURE-BASED CONTENT MODEL
# ----------------------------------------------------------

num_cols2 = ['course_duration_hours','course_price','feedback_score']
cat_cols2 = ['difficulty_level','certification_offered','study_material_available']

enc = OneHotEncoder()
encoded = enc.fit_transform(df_clean[cat_cols2]).toarray()

feature_matrix = pd.concat([
    df_clean[num_cols2].reset_index(drop=True),
    pd.DataFrame(encoded)
], axis=1)

feature_matrix.columns = feature_matrix.columns.astype(str)

scaler = StandardScaler()
feature_scaled = scaler.fit_transform(feature_matrix)


def get_similar_courses(course_name, top_n=5):
    try:
        idx = df_clean[df_clean["course_name"].str.lower() == course_name.lower()].index[0]
    except:
        return None
    sims = cosine_similarity(feature_scaled[idx].reshape(1,-1), feature_scaled).flatten()
    top_idx = sims.argsort()[::-1][1:top_n+1]
    return df_clean.iloc[top_idx][["course_id","course_name","course_price"]]


# ----------------------------------------------------------
# 3. TF-IDF TEXT MODEL
# ----------------------------------------------------------

df_clean["combined_text"] = (
    df_clean["course_name"].astype(str) + " " +
    df_clean["difficulty_level"].astype(str) + " " +
    df_clean["instructor"].astype(str)
)

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(df_clean["combined_text"])


def recommend_tfidf(course_name, top_n=5):
    course_name = course_name.lower()
    if course_name not in df_clean["course_name"].str.lower().values:
        return None
    
    idx = df_clean[df_clean["course_name"].str.lower() == course_name].index[0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][1:top_n+1]
    
    return df_clean.iloc[top_idx][["course_id","course_name","course_price"]]


# ----------------------------------------------------------
# 4. USER-BASED COLLABORATIVE FILTERING
# ----------------------------------------------------------

rating_matrix = df.pivot_table(index="user_id", columns="course_id", values="rating")
rating_filled = rating_matrix.fillna(0).values


def recommend_user_based(user_id, top_k=10, top_n=5):

    if user_id not in rating_matrix.index:
        return None

    target = rating_matrix.loc[user_id].fillna(0).values
    users = rating_matrix.index.tolist()

    sims = []
    for i, uid in enumerate(users):
        vec = rating_filled[i]
        sim = np.dot(target, vec) / (norm(target)*norm(vec) + 1e-8)
        sims.append((uid, sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    sims = [u for u in sims if u[0] != user_id][:top_k]

    scores = {}
    for (uid, sim) in sims:
        for _, r in df[df["user_id"] == uid][["course_id","rating"]].iterrows():
            cid = r["course_id"]
            scores[cid] = scores.get(cid, 0) + sim * r["rating"]

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = pd.DataFrame(ranked, columns=["course_id","score"])
    return result.merge(df_clean[["course_id","course_name"]].drop_duplicates(), on="course_id")


# ----------------------------------------------------------
# 5. RATING PREDICTION
# ----------------------------------------------------------

def predict_rating(user_id, course_id):

    if user_id not in rating_matrix.index:
        return None

    target = rating_matrix.loc[user_id].fillna(0).values
    users = rating_matrix.index.tolist()

    sims = []
    for i, uid in enumerate(users):
        vec = rating_filled[i]
        sim = np.dot(target, vec) / (norm(target)*norm(vec) + 1e-8)
        sims.append((uid, sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:15]

    num = den = 0
    for (uid, sim) in sims:
        rating = df[(df["user_id"] == uid) & (df["course_id"] == course_id)]["rating"]
        if not rating.empty:
            num += sim * rating.values[0]
            den += abs(sim)

    return num / (den + 1e-8)


# ----------------------------------------------------------
# 6. HYBRID MODEL (TF-IDF + Feature + User Model)
# ----------------------------------------------------------

def hybrid_recommender(course_name, user_id, top_n=5, 
                       w_tfidf=0.4, w_feature=0.4, w_user=0.2):

    # TF-IDF Model
    tfidf_df = recommend_tfidf(course_name, top_n=20)
    if tfidf_df is not None:
        tfidf_df["tfidf_score"] = np.linspace(1, 0.1, len(tfidf_df))
    else:
        tfidf_df = pd.DataFrame(columns=["course_id","tfidf_score"])

    # Feature model
    feat_df = get_similar_courses(course_name, top_n=20)
    if feat_df is not None:
        feat_df["feature_score"] = np.linspace(1, 0.1, len(feat_df))
    else:
        feat_df = pd.DataFrame(columns=["course_id","feature_score"])

    # User-based model
    user_df = recommend_user_based(user_id, top_n=20)
    if user_df is not None:
        user_df["user_score"] = np.linspace(1, 0.1, len(user_df))
    else:
        user_df = pd.DataFrame(columns=["course_id","user_score"])

    hybrid = pd.DataFrame({"course_id": df_clean["course_id"].unique()})
    hybrid = hybrid.merge(tfidf_df, on="course_id", how="left")
    hybrid = hybrid.merge(feat_df, on="course_id", how="left")
    hybrid = hybrid.merge(user_df, on="course_id", how="left")

    hybrid = hybrid.fillna(0)

    hybrid["final_score"] = (
        hybrid["tfidf_score"] * w_tfidf +
        hybrid["feature_score"] * w_feature +
        hybrid["user_score"] * w_user
    )

    top = hybrid.sort_values("final_score", ascending=False).head(top_n)

    return top.merge(
        df_clean[["course_id","course_name","course_price"]].drop_duplicates(),
        on="course_id",
        how="left"
    )


# ----------------------------------------------------------
# 7. STREAMLIT UI (Exactly like Screenshot UI)
# ----------------------------------------------------------

st.markdown(
    """
    <h1 style='text-align: center;'>🎓 Course Recommendation System</h1>
    <p style='text-align: center; color: gray;'>AI-powered personalized course recommendations</p>
    <hr>
    """,
    unsafe_allow_html=True
)

left, right = st.columns([1, 2])

# ---------------- LEFT PANEL (INPUT FORM) ----------------
with left:

    st.markdown("## ✨ Enter Your Preferences")

    user_id_input = st.selectbox(
        "User ID:", sorted(df["user_id"].unique())
    )

    course_selected = st.selectbox(
        "Choose a Course:", sorted(df_clean["course_name"].unique())
    )

    difficulty = st.selectbox(
        "Preferred Difficulty:", ["Beginner", "Intermediate", "Advanced"]
    )

    cert_required = st.radio("Certification Required?", ["Both", "Yes", "No"])

    mat_required = st.radio("Study Materials Required?", ["Both", "Yes", "No"])

    k_rec = st.slider("Number of Recommendations:", 1, 10, 5)

    submit = st.button("💡 Get Recommendations", use_container_width=True)


# ---------------- RIGHT PANEL (OUTPUT) ----------------
with right:

    if submit:

        st.success("Recommendations Generated Successfully! 🎉")

        st.markdown("### 🎯 Content-Based Similar Courses")
        st.write(get_similar_courses(course_selected, k_rec))

        st.markdown("### 📘 TF-IDF Similar Courses")
        st.write(recommend_tfidf(course_selected, k_rec))

        st.markdown("### 👥 User-Based Recommendations")
        st.write(recommend_user_based(user_id_input, k_rec))

        st.markdown("### 🔥 Hybrid Model Recommendations (Best Overall)")
        st.write(hybrid_recommender(course_selected, user_id_input, k_rec))

        st.markdown("### ⭐ Predicted Rating for Selected Course")
        cid = df_clean[df_clean["course_name"] == course_selected]["course_id"].iloc[0]
        st.write(predict_rating(user_id_input, cid))


# # ----------------------------------------------------------
# # 8. VISUALIZATION
# # ----------------------------------------------------------

# st.markdown("---")
# st.subheader("📊 Rating Distribution")

# fig, ax = plt.subplots(figsize=(7,4))
# sns.histplot(df["rating"], kde=True, ax=ax)
# st.pyplot(fig)






# import streamlit as st
# import pandas as pd
# import numpy as np
# from numpy.linalg import norm
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity

# # ---------------------------------------------------------
# # 1. LOAD & CLEAN DATA
# # ---------------------------------------------------------

# DATA_PATH = r"D:\Recommendation_system\online_course_recommendation_v2.csv"
# df = pd.read_csv(DATA_PATH)

# df = df.drop_duplicates()
# df = df.dropna(subset=["course_name", "user_id", "course_id"])

# num_cols = df.select_dtypes(include=["int64", "float64"]).columns
# cat_cols = df.select_dtypes(include=["object"]).columns

# df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
# df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# df_clean = df.copy()

# # ---------------------------------------------------------
# # 2. CONTENT-BASED ENGINEERING
# # ---------------------------------------------------------

# num_cols2 = ['course_duration_hours','course_price','feedback_score','rating']
# cat_cols2 = ['difficulty_level','study_material_available','certification_offered']

# enc = OneHotEncoder()
# encoded_cats = enc.fit_transform(df_clean[cat_cols2]).toarray()

# feature_matrix = pd.concat([
#     df_clean[num_cols2].reset_index(drop=True),
#     pd.DataFrame(encoded_cats)
# ], axis=1)

# feature_matrix.columns = feature_matrix.columns.astype(str)
# scaler = StandardScaler()
# feature_scaled = scaler.fit_transform(feature_matrix)

# def content_based(course_name, top_n=5):
#     try:
#         idx = df_clean[df_clean['course_name'].str.lower() == course_name.lower()].index[0]
#     except:
#         return None

#     sims = cosine_similarity(feature_scaled[idx].reshape(1, -1), feature_scaled).flatten()
#     top_idx = sims.argsort()[::-1][1:top_n+1]

#     result = df_clean.iloc[top_idx][[
#         'course_id', 'course_name', 'instructor', 'difficulty_level',
#         'course_price', 'rating', 'certification_offered', 'study_material_available'
#     ]]
#     return result.sort_values(by="rating", ascending=False)


# # ---------------------------------------------------------
# # 3. TF-IDF CONTENT MODEL
# # ---------------------------------------------------------

# df_clean["combined_text"] = (
#     df_clean["course_name"] + " " +
#     df_clean["difficulty_level"] + " " +
#     df_clean["instructor"]
# )

# tfidf = TfidfVectorizer(stop_words="english", max_features=4000)
# tfidf_matrix = tfidf.fit_transform(df_clean["combined_text"])

# def tfidf_recommend(course_name, top_n=5):
#     cname = course_name.lower()
#     if cname not in df_clean["course_name"].str.lower().values:
#         return None

#     idx = df_clean[df_clean["course_name"].str.lower() == cname].index[0]
#     sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
#     top_idx = sims.argsort()[::-1][1:top_n+1]

#     result = df_clean.iloc[top_idx][[
#         'course_id', 'course_name', 'instructor', 'difficulty_level',
#         'course_price', 'rating', 'certification_offered', 'study_material_available'
#     ]]
#     return result.sort_values(by="rating", ascending=False)


# # ---------------------------------------------------------
# # 4. LIGHT COLLABORATIVE FILTERING (SAFE)
# # ---------------------------------------------------------

# rating_df = df[['user_id', 'course_id', 'rating']]

# def user_based(user_id, top_n=5):
#     user_data = rating_df[rating_df['user_id'] == user_id]

#     if user_data.empty:
#         return None

#     similar_users = rating_df[rating_df['course_id'].isin(user_data['course_id'])]
#     similar_users = similar_users[similar_users['user_id'] != user_id]

#     scores = similar_users.groupby("course_id")["rating"].mean()

#     taken = set(user_data['course_id'])
#     scores = scores[~scores.index.isin(taken)]

#     top_courses = scores.sort_values(ascending=False).head(top_n)

#     result = df_clean[df_clean['course_id'].isin(top_courses.index)][[
#         'course_id', 'course_name', 'instructor', 'difficulty_level',
#         'course_price', 'rating', 'certification_offered','study_material_available'
#     ]]

#     return result.sort_values(by="rating", ascending=False)


# # ---------------------------------------------------------
# # 5. STREAMLIT UI — EXACT SCREENSHOT DESIGN
# # ---------------------------------------------------------

# st.set_page_config(page_title="Course Recommendation System", layout="wide")

# st.markdown("""
# <h1 style='text-align:center; font-size:42px;'>🎓 Course Recommendation System</h1>
# <p style='text-align:center; font-size:20px; color:grey;'>
# AI-powered recommendations tailored to your learning needs.
# </p>
# <hr>
# """, unsafe_allow_html=True)

# left, right = st.columns([1, 3])

# with left:
#     st.markdown("### ⚙ Enter Your Preferences")

#     user_id = st.selectbox("User ID:", sorted(df_clean["user_id"].unique()))

#     course_name = st.selectbox("Choose a Course:", sorted(df_clean["course_name"].unique()))

#     difficulty = st.selectbox("Preferred Difficulty:", sorted(df_clean["difficulty_level"].unique()))

#     cert_required = st.radio("Certification Required?", ["Both", "Yes", "No"])

#     material_required = st.radio("Study Material Required?", ["Both", "Yes", "No"])

#     top_n = st.slider("Number of Recommendations:", 3, 15, 6)

#     get_reco = st.button("🔍 Get Recommendations")


    

# if get_reco:
#     right.success("Recommendations Generated! 🎉")

#     right.markdown(f"""
#     <h2 style="font-size:30px;">🎯 Top {top_n} Personalized Recommendations</h2>
#     <p style='font-size:18px;'>
#     Based on <b>{course_name}</b>, category <b>{course_name.split()[0]}</b>,
#     difficulty <b>{difficulty}</b>, certification <b>{cert_required}</b>,
#     study material <b>{material_required}</b>.
#     </p>
#     <hr>
#     """, unsafe_allow_html=True)

#     # — Hybrid model —
#     cb = content_based(course_name, top_n)
#     tfidf_rec = tfidf_recommend(course_name, top_n)
#     ub = user_based(user_id, top_n)

#     combined = pd.concat([cb, tfidf_rec, ub]).drop_duplicates()

#     # Filters
#     if cert_required != "Both":
#         combined = combined[combined["certification_offered"] == cert_required]

#     if material_required != "Both":
#         combined = combined[combined["study_material_available"] == material_required]

#     combined = combined.sort_values(by="rating", ascending=False).head(top_n)

#     # Display cards
#     for _, row in combined.iterrows():
#         right.markdown(f"""
#         <div style="border:2px solid #4B9CD3; border-radius:15px; padding:18px;
#                     margin-bottom:20px; background:#F7FBFF;">
#             <h3>📘 {row['course_name']}</h3>
#             <p><b>Instructor:</b> {row['instructor']}</p>
#             <p><b>Difficulty:</b> {row['difficulty_level']}</p>
#             <p><b>Price:</b> ₹{row['course_price']}</p>
#             <p><b>Rating:</b> ⭐ {row['rating']} / 5</p>
#         </div>
#         """, unsafe_allow_html=True)



# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------------------------------------
# # BASIC PAGE SETUP
# # -----------------------------------------------------------
# st.set_page_config(
#     page_title="Course Recommendation System",
#     layout="wide"
# )





# # -----------------------------------------------------------
# # CUSTOM CSS — EXACT UI LIKE YOUR SCREENSHOT
# # -----------------------------------------------------------
# st.markdown("""
# <style>

# * { font-family: 'Poppins', sans-serif; }

# /* SIDEBAR BACKGROUND */
# section[data-testid="stSidebar"] {
#     background-color: #f5f6f8 !important;
#     padding: 20px;
# }

# /* Increase sidebar width */
# section[data-testid="stSidebar"] {
#     width: 400px !important;       /* change to any width you want */
#     min-width: 400px !important;
#      padding-top: -2px !important;      /* reduce from default ~60px */
#     margin-top: -1px !important;
#     aria-expanded="true"
#     style="position: relative; user-select: auto; width: 499px; height: auto; box-sizing: border-box; flex-shrink: 0;"
# }


# /* SIDEBAR SCROLL */
# section[data-testid="stSidebar"] > div {
#     height: 100vh;
#     overflow-y: auto;
# }

# /* RIGHT PANEL SCROLL */
# .main-block {
#     height: 5vh;
#     overflow-y: auto;
#     padding-right: 25px;
# }

# /* TITLE UNDERLINE */
# .sidebar-title {
#     font-size: 22px;
#     font-weight: 700;
#     border-bottom: 2px solid #3b82f6;
#     padding-bottom: 8px;
#     margin-bottom: 20px;
# }

# /* INPUT BOX SHAPE */
# div[data-baseweb="select"] > div {
#     border-radius: 10px !important;
#     background: #eef1f6 !important;
# }

# /* RADIO BUTTONS */
# .stRadio label {
#     font-size: 16px;
# }

# /* RED SLIDER THUMB */
# [data-testid="stSlider"] input[type="range"]::-webkit-slider-thumb {
#     background: #ef4444 !important;
# }

# /* GREEN BUTTON */
# .stButton > button {
#     width: 100%;
#     border-radius: 40px;
#     padding: 12px 20px;
#     font-size: 18px;
#     font-weight: 700;
#     border: none;
#     background: linear-gradient(135deg, #22c55e, #16a34a);
#     color: white;
# }
# .stButton > button:hover {
#     background: #15803d;
# }
            

# # div.block-container {
# #     padding-top: 0rem !important;
# #     margin-top: -3rem !important;
# # }


# </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------------------------
# # LOAD DATA
# # -----------------------------------------------------------
# DATA_PATH = r"D:\Recommendation_system\online_course_recommendation_v2.csv"
# df = pd.read_csv(DATA_PATH)

# df = df.drop_duplicates()
# df = df.dropna(subset=["course_name", "user_id", "course_id"])

# # FIX COLUMNS FOR ENCODER
# df.columns = df.columns.astype(str)
# df_clean = df.copy()

# # -----------------------------------------------------------
# # CONTENT-BASED FEATURES
# # -----------------------------------------------------------
# num_cols2 = ['course_duration_hours','course_price','feedback_score','rating']
# cat_cols2 = ['difficulty_level','study_material_available','certification_offered']

# enc = OneHotEncoder()
# encoded = enc.fit_transform(df_clean[cat_cols2].astype(str)).toarray()

# feature_matrix = pd.concat([
#     df_clean[num_cols2].reset_index(drop=True),
#     pd.DataFrame(encoded)
# ], axis=1)

# feature_matrix.columns = feature_matrix.columns.astype(str)

# scaler = StandardScaler()
# feature_scaled = scaler.fit_transform(feature_matrix)

# def content_based(course_name, top_n=5):
#     cname = course_name.lower()
#     if cname not in df_clean["course_name"].str.lower().values:
#         return None

#     idx = df_clean[df_clean["course_name"].str.lower() == cname].index[0]
#     sims = cosine_similarity(feature_scaled[idx].reshape(1,-1), feature_scaled).flatten()
#     top_idx = sims.argsort()[::-1][1:top_n+1]

#     return df_clean.iloc[top_idx]


# # -----------------------------------------------------------
# # TF-IDF MODEL
# # -----------------------------------------------------------
# df_clean["combined_text"] = (
#     df_clean["course_name"] + " " +
#     df_clean["difficulty_level"] + " " +
#     df_clean["instructor"]
# )

# tfidf = TfidfVectorizer(stop_words="english")
# tfidf_matrix = tfidf.fit_transform(df_clean["combined_text"])

# def tfidf_recommend(course_name, top_n=5):
#     cname = course_name.lower()
#     if cname not in df_clean["course_name"].str.lower().values:
#         return None
    
#     idx = df_clean[df_clean["course_name"].str.lower()==cname].index[0]
#     sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
#     top_idx = sims.argsort()[::-1][1:top_n+1]

#     return df_clean.iloc[top_idx]


# # -----------------------------------------------------------
# # COLLABORATIVE FILTERING
# # -----------------------------------------------------------
# rating_df = df[['user_id','course_id','rating']]

# def user_based(user_id, top_n=5):

#     user_data = rating_df[rating_df['user_id']==user_id]
#     if user_data.empty: return None

#     similar = rating_df[rating_df['course_id'].isin(user_data['course_id'])]
#     similar = similar[similar['user_id'] != user_id]

#     scores = similar.groupby("course_id")["rating"].mean()
#     taken = set(user_data['course_id'])

#     scores = scores[~scores.index.isin(taken)]
#     top_ids = scores.sort_values(ascending=False).head(top_n).index

#     return df_clean[df_clean["course_id"].isin(top_ids)]

# # --------------------------------------------
# # SESSION STATE INITIALIZATION
# # --------------------------------------------
# if "show_results" not in st.session_state:
#     st.session_state.show_results = False

# if "results" not in st.session_state:
#     st.session_state.results = None



# # -----------------------------------------------------------
# # SIDEBAR — EXACT LIKE SCREENSHOT
# # -----------------------------------------------------------



# with st.sidebar:

#     st.markdown("<p class='sidebar-title'>⚙ Enter Your Preferences</p>", unsafe_allow_html=True)

#     user_id = st.selectbox("User ID:", sorted(df_clean["user_id"].unique()))

#     course_name = st.selectbox("Choose a Course:", sorted(df_clean["course_name"].unique()))

#     difficulty = st.selectbox("Preferred Difficulty:", sorted(df_clean["difficulty_level"].unique()))

#     cert_required = st.radio("Certification Required?", ["Both","Yes","No"])

#     material_required = st.radio("Study Material Required?", ["Both","Yes","No"])

#     top_n = st.slider("Number of Recommendations:", 3, 15, 6)

#     if st.button("✨ Get Recommendations"):
#         st.session_state.show_results = True




# # -----------------------------------------------------------
# # RIGHT MAIN PANEL
# # -----------------------------------------------------------
# st.markdown("<div class='main-block'>", unsafe_allow_html=True)

# if st.session_state.show_results:

#     # MAIN HEADER EXACT LIKE SCREENSHOT
#     st.markdown("""
#     <h1 style='text-align:center; font-size:48px; font-weight:700; margin-top:-10px;'>
#         🎓 Course Recommendation System
#     </h1>
#     <p style='text-align:center; font-size:20px; color:#6b7280; margin-top:-15px;'>
#         AI-powered recommendations tailored to your learning needs.
#     </p>
#     <hr style='margin-top:25px; margin-bottom:40px;'>
#     """, unsafe_allow_html=True)

#     # TOP RECOMMENDATION TITLE
#     st.markdown(f"""
#     <h2 style='font-size:25px; font-weight:700; margin-bottom:5px;'>
#         🎯 Top {top_n} Personalized Recommendations
#     </h2>

#     <p style='font-size:18px; color:#374151; margin-top:-5px;'>
#         Based on <b>{course_name}</b>, category <b>{course_name.split()[0]}</b>,
#         difficulty <b>{difficulty}</b>, certification <b>{cert_required}</b>,
#         study material <b>{material_required}</b>.
#     </p>

#     <hr style='margin-top:20px; margin-bottom:35px;'>
#     """, unsafe_allow_html=True)

#     # HYBRID RECOMMENDATIONS
#     cb = content_based(course_name, top_n)
#     tfidf_rec = tfidf_recommend(course_name, top_n)
#     ub = user_based(user_id, top_n)

#     combined = pd.concat([cb, tfidf_rec, ub]).drop_duplicates()

#     if cert_required != "Both":
#         combined = combined[combined["certification_offered"] == cert_required]

#     if material_required != "Both":
#         combined = combined[combined["study_material_available"] == material_required]

#     combined = combined.sort_values(by="rating", ascending=False).head(top_n)

#     # DISPLAY CARDS
#     for _, row in combined.iterrows():
#         card = f"""
# <div style="
#     background:white;
#     border-left:6px solid #3b82f6;
#     border-radius:15px;
#     padding:22px;
#     margin-bottom:25px;
#     box-shadow:0px 4px 14px rgba(0,0,0,0.06);
#     transition:0.2s;
# "
# onmouseover="this.style.transform='translateY(-5px)';
#              this.style.boxShadow='0px 12px 25px rgba(0,0,0,0.15)'"
# onmouseout="this.style.transform='translateY(0px)';
#             this.style.boxShadow='0px 4px 14px rgba(0,0,0,0.06)'">

#     <!-- Course Icon & Title -->
#     <div style="font-size:20px; margin-bottom:-2px;">📖</div>
#     <h2 style="margin-top:5px; font-size:26px; font-weight:700; color:#1f2937;">
#         {row['course_name']}
#     </h2>

#     <!-- Instructor -->
#     <p style="font-size:16px; color:#374151; margin-top:6px;">
#         👨‍🏫 <b>{row['instructor']}</b>
#     </p>

#     <!-- BADGE ROW (Difficulty + Rating + Price) -->
#     <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:12px;">

#         <!-- Difficulty -->
#         <span style="
#             background:#e0f2fe;
#             color:#0369a1;
#             padding:6px 14px;
#             border-radius:20px;
#             font-weight:600;
#         ">
#             {row['difficulty_level']}
#         </span>

#         <!-- Rating -->
#         <span style="
#             background:#fef3c7;
#             color:#b45309;
#             padding:6px 14px;
#             border-radius:20px;
#             font-weight:600;
#         ">
#             ⭐ {row['rating']} / 5
#         </span>

#         <!-- Price Badge -->
#         <span style="
#             background:#dcfce7;
#             color:#166534;
#             padding:6px 14px;
#             border-radius:20px;
#             font-weight:600;
#         ">
#             💰 ₹{row['course_price']}
#         </span>
#     </div>

#     <!-- Certification -->
#     <p style="font-size:16px; color:#374151; margin-top:14px;">
#         🎓 <b>Certification:</b> ✓ {row['certification_offered']}
#     </p>

#     <!-- Study Material -->
#     <p style="font-size:16px; color:#374151; margin-top:-5px;">
#         📚 <b>Study Material:</b> ✓ {row['study_material_available']}
#     </p>

# </div>
# """


#         components.html(card, height=350)

# st.markdown("</div>", unsafe_allow_html=True)








# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------------------------------------
# # PAGE SETUP
# # -----------------------------------------------------------
# st.set_page_config(page_title="Course Recommendation System", layout="wide")

# # -----------------------------------------------------------
# # CUSTOM CSS
# # -----------------------------------------------------------
# st.markdown("""
# <style>
# * { font-family: 'Poppins', sans-serif; }

# /* SIDEBAR */
# section[data-testid="stSidebar"] {
#     background-color: #f5f6f8 !important;
#     padding: 20px;
#     width: 380px !important;
#     min-width: 380px !important;
# }
# section[data-testid="stSidebar"] > div {
#     height: 100vh;
#     overflow-y: auto;
# }

# /* RIGHT PANEL */
# .main-block {
#     height: 100vh;
#     overflow-y: auto;
#     padding-right: 20px;
# }

# /* INPUT BOXES */
# div[data-baseweb="select"] > div {
#     border-radius: 10px !important;
#     background: #eef1f6 !important;
# }

# /* BUTTON */
# .stButton > button {
#     width: 100%;
#     border-radius: 40px;
#     padding: 12px 20px;
#     font-size: 18px;
#     font-weight: 700;
#     border: none;
#     background: linear-gradient(135deg, #22c55e, #16a34a);
#     color: white;
# }
# .stButton > button:hover {
#     background: #15803d;
# }

# /* SLIDER */
# [data-testid="stSlider"] input[type="range"]::-webkit-slider-thumb {
#     background: #ef4444 !important;
# }

# </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------------------------
# # LOAD DATA
# # -----------------------------------------------------------
# DATA_PATH = r"D:\Recommendation_system\online_course_recommendation_v2.csv"
# df = pd.read_csv(DATA_PATH)

# df = df.drop_duplicates()
# df = df.dropna(subset=["course_name", "user_id", "course_id"])

# # make all column names string
# df.columns = df.columns.astype(str)
# df_clean = df.copy()

# # -----------------------------------------------------------
# # FEATURE ENGINEERING
# # -----------------------------------------------------------
# num_cols = ['course_duration_hours', 'course_price', 'feedback_score', 'rating']
# cat_cols = ['difficulty_level', 'study_material_available', 'certification_offered']

# enc = OneHotEncoder()
# encoded_cat = enc.fit_transform(df_clean[cat_cols].astype(str)).toarray()

# feature_matrix = pd.concat([
#     df_clean[num_cols].reset_index(drop=True),
#     pd.DataFrame(encoded_cat)
# ], axis=1)

# feature_matrix.columns = feature_matrix.columns.astype(str)

# scaler = StandardScaler()
# feature_scaled = scaler.fit_transform(feature_matrix)

# # -----------------------------------------------------------
# # MODELS
# # -----------------------------------------------------------
# def content_based(course_name, top_n):
#     cname = course_name.lower()
#     if cname not in df_clean["course_name"].str.lower().values:
#         return pd.DataFrame()

#     idx = df_clean[df_clean["course_name"].str.lower() == cname].index[0]
#     sims = cosine_similarity(feature_scaled[idx].reshape(1,-1), feature_scaled).flatten()
#     top_idx = sims.argsort()[::-1][1:top_n+1]
#     return df_clean.iloc[top_idx]


# df_clean["combined_text"] = (
#     df_clean["course_name"] + " " +
#     df_clean["difficulty_level"] + " " +
#     df_clean["instructor"]
# )

# tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
# tfidf_matrix = tfidf.fit_transform(df_clean["combined_text"])

# def tfidf_recommend(course_name, top_n):
#     cname = course_name.lower()
#     if cname not in df_clean["course_name"].str.lower().values:
#         return pd.DataFrame()

#     idx = df_clean[df_clean["course_name"].str.lower() == cname].index[0]
#     sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
#     top_idx = sims.argsort()[::-1][1:top_n+1]
#     return df_clean.iloc[top_idx]


# rating_df = df[['user_id', 'course_id', 'rating']]

# def user_based(user_id, top_n):
#     user_data = rating_df[rating_df["user_id"] == user_id]
#     if user_data.empty:
#         return pd.DataFrame()

#     similar = rating_df[rating_df["course_id"].isin(user_data["course_id"])]
#     similar = similar[similar['user_id'] != user_id]

#     scores = similar.groupby("course_id")["rating"].mean()
#     taken = set(user_data["course_id"])
#     scores = scores[~scores.index.isin(taken)]

#     top_ids = scores.sort_values(ascending=False).head(top_n).index
#     return df_clean[df_clean["course_id"].isin(top_ids)]

# # -----------------------------------------------------------
# # SESSION STATE
# # -----------------------------------------------------------
# if "show_results" not in st.session_state:
#     st.session_state.show_results = False

# # -----------------------------------------------------------
# # SIDEBAR UI
# # -----------------------------------------------------------
# with st.sidebar:

#     st.markdown("<p class='sidebar-title'>⚙ Enter Your Preferences</p>", unsafe_allow_html=True)

#     user_id = st.selectbox("User ID:", sorted(df_clean["user_id"].unique()))

#     course_name = st.selectbox("Choose a Course:", sorted(df_clean["course_name"].unique()))

#     # ⭐ NEW: Area of Interest
#     interests = st.multiselect(
#         "Area of Interest (Select one or more):",
#         options=sorted(df_clean["category"].unique()),
#         default=[]
#     )

#     difficulty = st.selectbox("Preferred Difficulty:", sorted(df_clean["difficulty_level"].unique()))

#     cert_required = st.radio("Certification Required?", ["Both","Yes","No"])

#     material_required = st.radio("Study Material Required?", ["Both","Yes","No"])

#     top_n = st.slider("Number of Recommendations:", 3, 15, 6)

#     if st.button("✨ Get Recommendations"):
#         st.session_state.show_results = True

# # -----------------------------------------------------------
# # RIGHT PANEL
# # -----------------------------------------------------------
# st.markdown("<div class='main-block'>", unsafe_allow_html=True)

# if st.session_state.show_results:

#     # TITLE SECTION
#     st.markdown("""
#     <h1 style='text-align:center; font-size:48px; font-weight:700;'>
#         🎓 Course Recommendation System
#     </h1>
#     <p style='text-align:center; font-size:20px; color:#6b7280;'>
#         AI-powered recommendations tailored to your learning needs.
#     </p>
#     <hr>
#     """, unsafe_allow_html=True)

#     # SUBTITLE
#     st.markdown(f"""
#         <h2>🎯 Top {top_n} Personalized Recommendations</h2>
#         <p>
#             Based on <b>{course_name}</b>, difficulty <b>{difficulty}</b>,
#             certification <b>{cert_required}</b>, study material <b>{material_required}</b>.
#         </p>
#         <hr>
#     """, unsafe_allow_html=True)

#     # 1️⃣ HYBRID RECOMMENDATION
#     cb = content_based(course_name, top_n)
#     tfidf_rec = tfidf_recommend(course_name, top_n)
#     ub = user_based(user_id, top_n)

#     combined = pd.concat([cb, tfidf_rec, ub]).drop_duplicates()

#     # 2️⃣ FILTER — Area of Interest
#     if interests:
#         combined = combined[combined["category"].isin(interests)]

#     # 3️⃣ SMART DIFFICULTY FILTER
#     difficulty_map = {
#         "Beginner": ["Beginner", "Intermediate"],
#         "Intermediate": ["Intermediate", "Beginner", "Advanced"],
#         "Advanced": ["Advanced", "Intermediate"]
#     }
#     combined = combined[combined["difficulty_level"].isin(difficulty_map[difficulty])]

#     # 4️⃣ CERTIFICATION + MATERIAL FILTERS
#     if cert_required != "Both":
#         combined = combined[combined["certification_offered"] == cert_required]

#     if material_required != "Both":
#         combined = combined[combined["study_material_available"] == material_required]

#     # 5️⃣ SORT
#     combined = combined.sort_values(by="rating", ascending=False).head(top_n)

#     # 🔥 IF NOTHING FOUND
#     if combined.empty:
#         st.warning("No courses match your exact preferences. Try relaxing one of the filters.")
#     else:
#         # DISPLAY CARDS
#         for _, row in combined.iterrows():
#             card = f"""
#             <div style="
#                 background:white; padding:22px; border-radius:15px;
#                 border-left:6px solid #3b82f6;
#                 margin-bottom:25px; box-shadow:0 4px 12px rgba(0,0,0,0.1);
#                 transition:0.2s;"
#                 onmouseover="this.style.transform='translateY(-6px)';
#                              this.style.boxShadow='0 10px 25px rgba(0,0,0,0.2)'"
#                 onmouseout="this.style.transform='translateY(0)';
#                             this.style.boxShadow='0 4px 12px rgba(0,0,0,0.1)'">

#                 <h2>📘 {row['course_name']}</h2>
#                 <p>👨‍🏫 <b>{row['instructor']}</b></p>

#                 <p><b>Difficulty:</b> {row['difficulty_level']}</p>
#                 <p><b>Rating:</b> ⭐ {row['rating']} / 5</p>
#                 <p><b>Price:</b> ₹{row['course_price']}</p>

#                 <p><b>Certification:</b> ✓ {row['certification_offered']}</p>
#                 <p><b>Study Material:</b> ✓ {row['study_material_available']}</p>
#             </div>
#             """
#             components.html(card, height=350)

# st.markdown("</div>", unsafe_allow_html=True)







# import streamlit as st
# import streamlit.components.v1 as components
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------------------------------------
# # PAGE SETUP
# # -----------------------------------------------------------
# st.set_page_config(page_title="Course Recommendation System", layout="wide")

# # -----------------------------------------------------------
# # CUSTOM CSS (same look as before)
# # -----------------------------------------------------------
# st.markdown("""
# <style>
# * { font-family: 'Poppins', sans-serif; }

# /* SIDEBAR */
# section[data-testid="stSidebar"] {
#     background-color: #f5f6f8 !important;
#     padding: 20px;
#     width: 380px !important;
#     min-width: 380px !important;
#     margin-bottom: 1rem;
#     height: 0.25rem;
# }
# section[data-testid="stSidebar"] > div {
#     height: 100vh;
#     overflow-y: auto;
# }

# /* RIGHT PANEL */
# .main-block {
#     height: 1vh; 
#     overflow-y: auto;
#     padding-right: 20px;
# }

# /* INPUT BOXES */
# div[data-baseweb="select"] > div {
#     border-radius: 10px !important;
#     background: #eef1f6 !important;
# }

# /* BUTTON */
# .stButton > button {
#     width: 100%;
#     border-radius: 40px;
#     padding: 12px 20px;
#     font-size: 18px;
#     font-weight: 700;
#     border: none;
#     background: linear-gradient(135deg, #22c55e, #16a34a);
#     color: white;
# }
# .stButton > button:hover {
#     background: #15803d;
# }

# /* SLIDER */
# [data-testid="stSlider"] input[type="range"]::-webkit-slider-thumb {
#     background: #ef4444 !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------------------------
# # LOAD DATA
# # -----------------------------------------------------------
# DATA_PATH = r"D:\Recommendation_system\online_course_recommendation_v2.csv"
# df = pd.read_csv(DATA_PATH)

# # basic cleaning
# df = df.drop_duplicates()
# df = df.dropna(subset=["course_name", "user_id", "course_id"])

# # make all column names string
# df.columns = df.columns.astype(str)
# df_clean = df.copy()

# # -----------------------------------------------------------
# # PREPARE FEATURES FOR SIMILARITY
# # -----------------------------------------------------------

# # numeric & categorical feature lists
# num_cols = ['course_duration_hours', 'course_price', 'feedback_score', 'rating']
# cat_cols = ['difficulty_level', 'study_material_available', 'certification_offered']

# # fill numeric/categorical NaNs if any
# df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
# df_clean[cat_cols] = df_clean[cat_cols].fillna("Unknown")

# # one-hot encode categorical
# enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
# encoded_cat = enc.fit_transform(df_clean[cat_cols].astype(str))

# encoded_df = pd.DataFrame(
#     encoded_cat,
#     columns=enc.get_feature_names_out(cat_cols),
#     index=df_clean.index
# )

# # combine numeric + encoded categorical
# feature_matrix = pd.concat(
#     [df_clean[num_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)],
#     axis=1
# )

# # scale
# scaler = StandardScaler()
# feature_scaled = scaler.fit_transform(feature_matrix)

# # -----------------------------------------------------------
# # TF-IDF TEXT FEATURES (for course title + difficulty + instructor)
# # -----------------------------------------------------------
# df_clean["combined_text"] = (
#     df_clean["course_name"].astype(str) + " " +
#     df_clean["difficulty_level"].astype(str) + " " +
#     df_clean["instructor"].astype(str)
# )

# tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
# tfidf_matrix = tfidf.fit_transform(df_clean["combined_text"])

# # -----------------------------------------------------------
# # NORMALISED YES/NO FOR FILTERS
# # -----------------------------------------------------------
# df_clean["cert_norm"] = df_clean["certification_offered"].astype(str).str.strip().str.lower()
# df_clean["study_norm"] = df_clean["study_material_available"].astype(str).str.strip().str.lower()

# YES_VALUES = {"yes", "1", "true", "y"}
# NO_VALUES = {"no", "0", "false", "n"}

# # -----------------------------------------------------------
# # HYBRID RECOMMENDER
# #   1) Find similar courses to selected course (features + text)
# #   2) Filter by difficulty, certification, study material
# #   3) Sort by rating (desc) then similarity (desc)
# # -----------------------------------------------------------
# def hybrid_recommend(course_name, difficulty_choice,
#                      cert_filter, material_filter, top_n=6):

#     # --- find index of selected course (exact match) ---
#     mask = df_clean["course_name"].str.lower() == course_name.lower()
#     if not mask.any():
#         return pd.DataFrame()  # no such course

#     base_idx = df_clean[mask].index[0]

#     # --- similarity from numeric + categorical features ---
#     feat_sims = cosine_similarity(
#         feature_scaled[base_idx].reshape(1, -1),
#         feature_scaled
#     ).flatten()

#     # --- text similarity from TF-IDF ---
#     text_sims = cosine_similarity(
#         tfidf_matrix[base_idx],
#         tfidf_matrix
#     ).flatten()

#     # --- hybrid similarity ---
#     hybrid_sims = 0.5 * feat_sims + 0.5 * text_sims

#     rec_df = df_clean.copy()
#     rec_df["similarity"] = hybrid_sims

#     # remove the selected course itself
#     rec_df = rec_df[rec_df.index != base_idx]

#     # 1️⃣ strict difficulty filter (exact level)
#     rec_df = rec_df[rec_df["difficulty_level"] == difficulty_choice]

#     # 2️⃣ certification filter
#     if cert_filter != "Both":
#         if cert_filter == "Yes":
#             rec_df = rec_df[rec_df["cert_norm"].isin(YES_VALUES)]
#         else:  # "No"
#             rec_df = rec_df[rec_df["cert_norm"].isin(NO_VALUES)]

#     # 3️⃣ study material filter
#     if material_filter != "Both":
#         if material_filter == "Yes":
#             rec_df = rec_df[rec_df["study_norm"].isin(YES_VALUES)]
#         else:  # "No"
#             rec_df = rec_df[rec_df["study_norm"].isin(NO_VALUES)]

#     # 4️⃣ sort: rating highest first, then similarity
# # 4️⃣ Weighted score: 70% similarity + 30% rating
#     if rec_df.empty:
#         return rec_df

# # calculate weighted score
#     rec_df["score"] = (0.6 * rec_df["similarity"]) + (0.4 * rec_df["rating"])

# # sort by score
#     rec_df = rec_df.sort_values(
#         by="score",
#         ascending=False
#     ).head(top_n)

#     return rec_df


# # -----------------------------------------------------------
# # SESSION STATE
# # -----------------------------------------------------------
# if "show_results" not in st.session_state:
#     st.session_state.show_results = False

# # -----------------------------------------------------------
# # SIDEBAR UI (exactly like your screenshot)
# # -----------------------------------------------------------
# with st.sidebar:
#     st.header("⚙ Enter Your Preferences")

#     user_id = st.selectbox("User ID:", sorted(df_clean["user_id"].unique().tolist()))

#     course_name = st.selectbox("Choose a Course:", sorted(df_clean["course_name"].unique().tolist()))

#     difficulty = st.selectbox("Preferred Difficulty:", sorted(df_clean["difficulty_level"].unique().tolist()))

#     cert_required = st.radio("Certification Required?", ["Both", "Yes", "No"])

#     material_required = st.radio("Study Material Required?", ["Both", "Yes", "No"])

#     top_n = st.slider("Number of Recommendations:", 3, 15, 6)

#     if st.button("✨ Get Recommendations"):
#         st.session_state.show_results = True

# # -----------------------------------------------------------
# # RIGHT PANEL
# # -----------------------------------------------------------
# st.markdown("<div class='main-block'>", unsafe_allow_html=True)

# if st.session_state.show_results:
#     # Title
#     st.markdown("""
#     <h1 style='text-align:center; font-size:48px; font-weight:700;'>
#         🎓 Course Recommendation System
#     </h1>
#     <p style='text-align:center; font-size:20px; color:#6b7280;'>
#         AI-powered recommendations tailored to your learning needs.
#     </p>
#     <hr>
#     """, unsafe_allow_html=True)

#     # Subtitle / filter summary
#     st.markdown(f"""
#         <h2>🎯 Top {top_n} Personalized Recommendations</h2>
#         <p>
#             Filtering strictly by:
#             difficulty <b>{difficulty}</b>,
#             certification <b>{cert_required}</b>,
#             study material <b>{material_required}</b>.
#         </p>
#         <hr>
#     """, unsafe_allow_html=True)

#     # Get recommendations
#     recommendations = hybrid_recommend(
#         course_name=course_name,
#         difficulty_choice=difficulty,
#         cert_filter=cert_required,
#         material_filter=material_required,
#         top_n=top_n
#     )

#     if recommendations.empty:
#         st.warning("No courses match your exact preferences. Try relaxing one of the filters.")
#     else:
#         # Show cards
#         for _, row in recommendations.iterrows():
#             cert_value = "✓ Yes" if str(row["cert_norm"]) in YES_VALUES else "✗ No"
#             mat_value = "✓ Yes" if str(row["study_norm"]) in YES_VALUES else "✗ No"

#             card = f"""
#             <div style="
#                 background:white; padding:22px; border-radius:15px;
#                 border-left:6px solid #3b82f6;
#                 margin-bottom:25px; box-shadow:0 4px 12px rgba(0,0,0,0.1);
#                 transition:0.2s;"
#                 onmouseover="this.style.transform='translateY(-6px)';
#                              this.style.boxShadow='0 10px 25px rgba(0,0,0,0.2)'"
#                 onmouseout="this.style.transform='translateY(0)';
#                             this.style.boxShadow='0 4px 12px rgba(0,0,0,0.1)'">

#                 <h2>📘 {row['course_name']}</h2>
#                 <p>👩‍🏫 <b>{row['instructor']}</b></p>

#                 <p><b>Difficulty:</b> {row['difficulty_level']}</p>
#                 <p><b>Rating:</b> ⭐ {row['rating']} / 5</p>
#                 <p><b>Price:</b> ₹{row['course_price']}</p>

#                 <p><b>Certification:</b> {cert_value}</p>
#                 <p><b>Study Material:</b> {mat_value}</p>
#             </div>
#             """
#             components.html(card, height=330)

# st.markdown("</div>", unsafe_allow_html=True)
