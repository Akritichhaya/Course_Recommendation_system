# import streamlit as st
# import pandas as pd
# import random
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------------------------------------
# # PAGE CONFIG
# # -----------------------------------------------------------
# st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# # -----------------------------------------------------------
# # CUSTOM CSS FOR BEAUTIFUL UI
# # -----------------------------------------------------------
# st.markdown("""
# <style>

# body {
#     background-color: #f4f6fa;
# }

# /* Card Design */
# .card {
#     background: white;
#     padding: 18px;
#     border-radius: 12px;
#     margin-bottom: 18px;
#     box-shadow: 0 4px 15px rgba(0,0,0,0.08);
#     border-left: 6px solid #4c83ff;
# }

# .card:hover {
#     transform: scale(1.01);
#     transition: 0.2s ease-in-out;
#     box-shadow: 0 8px 20px rgba(0,0,0,0.18);
# }

# /* Title */
# .top-header {
#     font-size: 45px;
#     font-weight: 800;
#     color: #1b263b;
#     margin-bottom: -10px;
# }

# .course-title {
#     font-size: 20px;
#     font-weight: 700;
#     color: #1b263b;
# }

# .instructor {
#     font-size: 15px;
#     font-weight: 600;
#     color: #4c83ff;
# }

# .difficulty {
#     font-size: 14px;
#     color: #696969;
# }

# .star-rating {
#     font-size: 18px;
#     color: #f4c430;
# }

# </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------------------------
# # LOAD DATA
# # -----------------------------------------------------------
# @st.cache_data
# def load_data():
#     df = pd.read_csv("online_course_recommendation_v2.csv")

#     df['certification_offered'] = df['certification_offered'].map({'Yes':1, 'No':0})
#     df['study_material_available'] = df['study_material_available'].map({'Yes':1, 'No':0})
#     df['difficulty_level_encoded'] = df['difficulty_level'].map({'Beginner':1, 'Intermediate':2, 'Advanced':3})

#     return df

# df = load_data()

# # -----------------------------------------------------------
# # TITLE
# # -----------------------------------------------------------
# st.markdown("<h1 class='top-header'>🎓 Course Recommendation System</h1>", unsafe_allow_html=True)
# st.write("AI-powered recommendations using **Surprise SVD Collaborative Filtering**.")
# st.write("---")

# # -----------------------------------------------------------
# # SIDEBAR SETTINGS
# # -----------------------------------------------------------
# st.sidebar.title("🔧 Settings")

# user_list = sorted(df['user_id'].unique())
# selected_user = st.sidebar.selectbox("Select User ID:", user_list)

# num_rec = st.sidebar.slider("Number of Recommendations:", 1, 10, 5)

# # -----------------------------------------------------------
# # BUILDING SVD MODEL
# # -----------------------------------------------------------
# reader = Reader(rating_scale=(0, 1))  
# data = Dataset.load_from_df(df[['user_id', 'course_id', 'rating']], reader)

# trainset, testset = train_test_split(data, test_size=0.2)

# model = SVD()
# model.fit(trainset)

# st.sidebar.success("✅ SVD Model Loaded Successfully!")

# # -----------------------------------------------------------
# # RECOMMENDATION FUNCTION (SVD BASED)
# # -----------------------------------------------------------
# def recommend_courses(user_id, n=5):
#     all_courses = df['course_id'].unique()
#     rated = df[df['user_id'] == user_id]['course_id'].unique()

#     # Courses user has NOT seen
#     remaining = [c for c in all_courses if c not in rated]

#     # Predict ratings
#     predictions = [(c, model.predict(user_id, c).est) for c in remaining]

#     predictions.sort(key=lambda x: x[1], reverse=True)

#     top_ids = [cid for cid, _ in predictions[:n]]

#     return df[df['course_id'].isin(top_ids)][[
#         'course_id', 'course_name', 'instructor', 'difficulty_level', 'rating'
#     ]].drop_duplicates()

# # -----------------------------------------------------------
# # USER DASHBOARD
# # -----------------------------------------------------------
# st.subheader("👤 User Profile Overview")

# user_data = df[df['user_id'] == selected_user]

# total_courses = len(user_data)
# avg_rating = round(user_data['rating'].mean() * 5, 2)
# total_time = round(user_data['time_spent_hours'].sum(), 2)
# difficulty_pref = user_data['difficulty_level'].mode()[0]
# fav_instructor = user_data['instructor'].mode()[0]

# st.markdown(
#     f"""
#     <div class="card">
#         <h3>📌 User Summary</h3>
#         <p><b>User ID:</b> {selected_user}</p>
#         <p><b>Total Courses Taken:</b> {total_courses}</p>
#         <p><b>Average Rating Given:</b> ⭐ {avg_rating}</p>
#         <p><b>Total Time Spent:</b> ⏳ {total_time} hrs</p>
#         <p><b>Preferred Difficulty:</b> {difficulty_pref}</p>
#         <p><b>Favorite Instructor:</b> 👨‍🏫 {fav_instructor}</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # -----------------------------------------------------------
# # SEARCH COURSES
# # -----------------------------------------------------------
# st.subheader("🔎 Search Courses")

# search_query = st.text_input("Search by Course Name:", "")

# if search_query.strip():
#     results = df[df['course_name'].str.contains(search_query, case=False)]
#     st.write(f"Found {len(results)} courses:")

#     for _, row in results.iterrows():
#         stars = "⭐" * int(round(row['rating'] * 5))
#         st.markdown(
#             f"""
#             <div class="card">
#                 <h4 class="course-title">{row['course_name']}</h4>
#                 <p class="instructor">👨‍🏫 {row['instructor']}</p>
#                 <p class="difficulty">📘 Difficulty: {row['difficulty_level']}</p>
#                 <p class="star-rating">{stars}</p>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

# st.write("---")

# # -----------------------------------------------------------
# # SHOW RECOMMENDATIONS
# # -----------------------------------------------------------
# if st.sidebar.button("Get Recommendations"):
#     st.subheader(f"🎯 Top {num_rec} Courses For You")

#     recs = recommend_courses(selected_user, num_rec)

#     for _, row in recs.iterrows():
#         img_url = f"https://source.unsplash.com/600x300/?education,learning,{random.randint(1,100)}"
#         stars = "⭐" * int(round(row['rating'] * 5))

#         st.markdown(
#             f"""
#             <div class="card">
#                 <img src="{img_url}" width="100%" style="border-radius:10px;">
#                 <h4 class="course-title">{row['course_name']}</h4>
#                 <p class="instructor">👨‍🏫 {row['instructor']}</p>
#                 <p class="difficulty">📘 Difficulty: {row['difficulty_level']}</p>
#                 <p class="star-rating">{stars}</p>
#                 <p><b>Course ID:</b> {row['course_id']}</p>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

# # -----------------------------------------------------------
# # SIMILAR COURSES (USING SVD LATENT FACTORS)
# # -----------------------------------------------------------
# st.subheader("📚 Find Similar Courses")

# df['cid_cat'] = df['course_id'].astype('category')
# course_index = {cid: i for i, cid in enumerate(df['cid_cat'].cat.categories)}

# def similar_courses(course_id, n=5):
#     item_factors = model.qi  
#     idx = course_index[course_id]

#     sims = cosine_similarity([item_factors[idx]], item_factors)[0]

#     top_idx = sims.argsort()[-n-1:-1][::-1]
#     similar_ids = [df['cid_cat'].cat.categories[i] for i in top_idx]

#     return df[df['course_id'].isin(similar_ids)][['course_id','course_name','rating']].drop_duplicates()

# selected_course = st.selectbox("Choose Course ID:", sorted(df['course_id'].unique()))

# if st.button("Show Similar Courses"):
#     sims = similar_courses(selected_course)

#     for _, row in sims.iterrows():
#         stars = "⭐" * int(round(row['rating'] * 5))
#         st.markdown(
#             f"""
#             <div class="card">
#                 <h4 class="course-title">{row['course_name']}</h4>
#                 <p class="star-rating">{stars}</p>
#                 <p><b>Course ID:</b> {row['course_id']}</p>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )




# import streamlit as st
# import pandas as pd
# import random
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------------------------------------
# # PAGE CONFIG
# # -----------------------------------------------------------
# st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# # -----------------------------------------------------------
# # CUSTOM CSS
# # -----------------------------------------------------------
# st.markdown("""
# <style>

# body {
#     background-color: #f4f6fa;
# }

# .card {
#     background: white;
#     padding: 18px;
#     border-radius: 12px;
#     margin-bottom: 18px;
#     box-shadow: 0 4px 15px rgba(0,0,0,0.08);
#     border-left: 6px solid #4c83ff;
# }

# .card:hover {
#     transform: scale(1.01);
#     transition: 0.2s ease-in-out;
#     box-shadow: 0 8px 20px rgba(0,0,0,0.18);
# }

# .top-header {
#     font-size: 45px;
#     font-weight: 800;
#     color: #1b263b;
# }

# .course-title {
#     font-size: 20px;
#     font-weight: 700;
#     color: #1b263b;
# }

# .instructor {
#     font-size: 15px;
#     font-weight: 600;
#     color: #4c83ff;
# }

# .difficulty {
#     font-size: 14px;
#     color: #696969;
# }

# .star-rating {
#     font-size: 18px;
#     color: #f4c430;
# }

# </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------------------------
# # LOAD DATA
# # -----------------------------------------------------------
# @st.cache_data
# def load_data():
#     df = pd.read_csv("online_course_recommendation_v2.csv")

#     df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0})
#     df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0})
#     df['difficulty_level_encoded'] = df['difficulty_level'].map({'Beginner': 1, 'Intermediate': 2, 'Advanced': 3})

#     return df

# df = load_data()

# # -----------------------------------------------------------
# # TITLE
# # -----------------------------------------------------------
# st.markdown("<h1 class='top-header'>🎓 Course Recommendation System</h1>", unsafe_allow_html=True)
# st.write("AI-powered recommendations using **Surprise SVD Collaborative Filtering**.")
# st.write("---")

# # -----------------------------------------------------------
# # EXPECTATIONS PANEL
# # -----------------------------------------------------------
# with st.expander("📘 What This Project Delivers (Required Expectations)"):
#     st.markdown("""
#     ### ✅ 1. Trained Recommendation Model (Backend)
#     - Collaborative Filtering (SVD)
#     - Content-Based Filtering
#     - Similarity using cosine distance

#     ### ✅ 2. Application Interface (Frontend)
#     - User ID input
#     - Course search
#     - Manual & dropdown course selection

#     ### ✅ 3. User Interaction Layer
#     - Inputs + Dynamic Outputs
#     - Course metadata & instructor info

#     ### ✅ 4. Real-Time Recommendation Flow
#     ```
#     User → Streamlit App → ML Model → Recommendations → User
#     ```

#     ### ✅ 5. Deployment
#     - Run using: `streamlit run app.py`
#     - Can be hosted on Streamlit Cloud / Heroku / AWS
#     """)

# st.write("---")

# # -----------------------------------------------------------
# # SIDEBAR SETTINGS
# # -----------------------------------------------------------
# st.sidebar.title("🔧 Settings")
# selected_user = st.sidebar.selectbox("Select User ID:", sorted(df['user_id'].unique()))
# num_rec = st.sidebar.slider("Number of Recommendations:", 1, 10, 5)

# # -----------------------------------------------------------
# # TRAIN SVD MODEL
# # -----------------------------------------------------------
# reader = Reader(rating_scale=(0, 1))
# data = Dataset.load_from_df(df[['user_id', 'course_id', 'rating']], reader)
# trainset, testset = train_test_split(data, test_size=0.2)

# model = SVD()
# model.fit(trainset)
# st.sidebar.success("✅ SVD Model Loaded Successfully!")

# # -----------------------------------------------------------
# # COURSE OVERVIEW + DROPDOWN
# # -----------------------------------------------------------
# st.subheader("📚 Course Overview")

# total_courses = df['course_name'].nunique()
# st.info(f"📘 Total Courses Available: **{total_courses}**")

# course_list = sorted(df['course_name'].unique())

# selected_course_dropdown = st.selectbox("Select a Course:", course_list)

# manual_input = st.text_input("Or Type a Course Name:")

# if manual_input:
#     if manual_input in course_list:
#         st.success(f"✔ '{manual_input}' is available.")
#         selected_course_dropdown = manual_input
#     else:
#         st.error(f"❌ '{manual_input}' is NOT available in the database.")

# selected_course_row = df[df['course_name'] == selected_course_dropdown].iloc[0]
# stars = "⭐" * int(round(selected_course_row['rating'] * 5))

# st.markdown(
#     f"""
#     <div class="card">
#         <h4 class="course-title">{selected_course_row['course_name']}</h4>
#         <p class="instructor">👨‍🏫 {selected_course_row['instructor']}</p>
#         <p class="difficulty">📘 Difficulty: {selected_course_row['difficulty_level']}</p>
#         <p class="star-rating">{stars}</p>
#         <p><b>Course ID:</b> {selected_course_row['course_id']}</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # -----------------------------------------------------------
# # USER PROFILE
# # -----------------------------------------------------------
# st.subheader("👤 User Profile Overview")

# user_data = df[df['user_id'] == selected_user]
# total_courses_taken = len(user_data)
# avg_rating = round(user_data['rating'].mean() * 5, 2)
# time_spent = round(user_data['time_spent_hours'].sum(), 2)
# difficulty_pref = user_data['difficulty_level'].mode()[0]
# fav_instructor = user_data['instructor'].mode()[0]

# st.markdown(
#     f"""
#     <div class="card">
#         <h3>📌 User Summary</h3>
#         <p><b>User ID:</b> {selected_user}</p>
#         <p><b>Total Courses Taken:</b> {total_courses_taken}</p>
#         <p><b>Average Rating:</b> ⭐ {avg_rating}</p>
#         <p><b>Total Time Spent:</b> ⏳ {time_spent} hrs</p>
#         <p><b>Preferred Difficulty:</b> {difficulty_pref}</p>
#         <p><b>Favourite Instructor:</b> {fav_instructor}</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# # -----------------------------------------------------------
# # RECOMMENDATION FUNCTION
# # -----------------------------------------------------------
# def recommend_courses(user_id, n=5):
#     all_courses = df['course_id'].unique()
#     rated = df[df['user_id'] == user_id]['course_id'].unique()
#     remaining = [c for c in all_courses if c not in rated]

#     preds = [(c, model.predict(user_id, c).est) for c in remaining]
#     preds.sort(key=lambda x: x[1], reverse=True)

#     top_ids = [cid for cid, _ in preds[:n]]

#     return df[df['course_id'].isin(top_ids)].drop_duplicates()

# # -----------------------------------------------------------
# # TOP RECOMMENDATIONS
# # -----------------------------------------------------------
# if st.sidebar.button("Get Recommendations"):
#     st.subheader(f"🎯 Top {num_rec} Courses For You")

#     recs = recommend_courses(selected_user, num_rec)

#     for _, row in recs.iterrows():
#         stars = "⭐" * int(round(row['rating'] * 5))
#         img = f"https://source.unsplash.com/600x300/?education,learning,{random.randint(1,100)}"

#         st.markdown(
#             f"""
#             <div class="card">
#                 <img src="{img}" width="100%" style="border-radius:10px;">
#                 <h4 class="course-title">{row['course_name']}</h4>
#                 <p class="instructor">👨‍🏫 {row['instructor']}</p>
#                 <p class="difficulty">📘 Difficulty: {row['difficulty_level']}</p>
#                 <p class="star-rating">{stars}</p>
#                 <p><b>Course ID:</b> {row['course_id']}</p>
#             </div>
#             """,
#             unsafe_allow_html=True
#         )

# # -----------------------------------------------------------
# # SIMILAR COURSES SECTION
# # -----------------------------------------------------------
# st.subheader("📚 Find Similar Courses")

# df['cid_cat'] = df['course_id'].astype('category')
# course_index = {cid: i for i, cid in enumerate(df['cid_cat'].cat.categories)}

# def similar_courses(course_id, n=5):
#     item_factors = model.qi
#     idx = course_index[course_id]
#     sims = cosine_similarity([item_factors[idx]], item_factors)[0]

#     top_idx = sims.argsort()[-n-1:-1][::-1]
#     sim_ids = [df['cid_cat'].cat.categories[i] for i in top_idx]

#     return df[df['course_id'].isin(sim_ids)].drop_duplicates()

# course_choice = st.selectbox("Choose Course ID to Find Similar:", sorted(df['course_id'].unique()))

# if st.button("Show Similar Courses"):
#     sims = similar_courses(course_choice)

#     for _, row in sims.iterrows():
#         stars = "⭐" * int(round(row['rating'] * 5))
#         st.success(f"**{row['course_name']}** — {stars} (ID: {row['course_id']})")



# import streamlit as st
# import pandas as pd
# import random
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------------------------------------
# # PAGE CONFIG
# # -----------------------------------------------------------
# st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# # -----------------------------------------------------------
# # CUSTOM CSS
# # -----------------------------------------------------------
# st.markdown("""
# <style>

# body {
#     background-color: #f4f6fa;
# }

# .card {
#     background: white;
#     padding: 18px;
#     border-radius: 12px;
#     margin-bottom: 18px;
#     box-shadow: 0 4px 15px rgba(0,0,0,0.08);
#     border-left: 6px solid #4c83ff;
# }

# .card:hover {
#     transform: scale(1.01);
#     transition: 0.2s ease-in-out;
#     box-shadow: 0 8px 20px rgba(0,0,0,0.18);
# }

# .top-header {
#     font-size: 45px;
#     font-weight: 800;
#     color: #1b263b;
# }

# .course-title {
#     font-size: 20px;
#     font-weight: 700;
#     color: #1b263b;
# }

# .instructor {
#     font-size: 15px;
#     font-weight: 600;
#     color: #4c83ff;
# }

# .difficulty {
#     font-size: 14px;
#     color: #696969;
# }

# .star-rating {
#     font-size: 18px;
#     color: #f4c430;
# }

# </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------------------------
# # LOAD DATA
# # -----------------------------------------------------------
# @st.cache_data
# def load_data():
#     df = pd.read_csv("online_course_recommendation_v2.csv")

#     df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0})
#     df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0})
#     df['difficulty_level_encoded'] = df['difficulty_level'].map({'Beginner': 1, 'Intermediate': 2, 'Advanced': 3})

#     return df

# df = load_data()

# # -----------------------------------------------------------
# # TITLE
# # -----------------------------------------------------------
# st.markdown("<h1 class='top-header'>🎓 Course Recommendation System</h1>", unsafe_allow_html=True)
# st.write("AI-powered recommendations using **Surprise SVD Collaborative Filtering + User Preferences**.")
# st.write("---")

# # -----------------------------------------------------------
# # EXPECTATIONS PANEL
# # -----------------------------------------------------------
# with st.expander("📘 What This Project Delivers"):
#     st.markdown("""
#     ### 🎯 INPUT Fields:
#     - User ID  
#     - User Interests (Domain)  
#     - Preferred Difficulty  
#     - Certification Requirement  
#     - Study Material Requirement  

#     ### 🎯 OUTPUT Fields:
#     - List of Recommended Courses  
#     - Instructor Name  
#     - Certification Availability  
#     - Study Material Availability  
#     """)

# st.write("---")

# # -----------------------------------------------------------
# # SIDEBAR SETTINGS (NEW INPUTS)
# # -----------------------------------------------------------
# st.sidebar.header("🔧 Input Settings")

# # USER ID
# selected_user = st.sidebar.selectbox("Select User ID:", sorted(df['user_id'].unique()))

# # NUMBER OF RECOMMENDATIONS
# num_rec = st.sidebar.slider("Number of Recommendations:", 1, 10, 5)

# # USER INTERESTS (CATEGORY)
# user_interest = st.sidebar.selectbox(
#     "Select Your Interest / Category:",
#     sorted(df['category'].unique()) if "category" in df.columns else ["Programming", "Data Science", "AI", "Business"]
# )

# # DIFFICULTY FILTER
# difficulty_choice = st.sidebar.multiselect(
#     "Preferred Difficulty:",
#     options=df['difficulty_level'].unique(),
#     default=df['difficulty_level'].unique()
# )

# # CERTIFICATION FILTER
# cert_choice = st.sidebar.radio(
#     "Certification Required?",
#     options=["Yes", "No", "Both"],
#     index=2
# )

# # STUDY MATERIAL FILTER
# material_choice = st.sidebar.radio(
#     "Study Material Required?",
#     options=["Yes", "No", "Both"],
#     index=2
# )

# # -----------------------------------------------------------
# # TRAIN MODEL (SVD)
# # -----------------------------------------------------------
# reader = Reader(rating_scale=(0, 1))
# data = Dataset.load_from_df(df[['user_id', 'course_id', 'rating']], reader)
# trainset, testset = train_test_split(data, test_size=0.2)

# model = SVD()
# model.fit(trainset)

# st.sidebar.success("✅ SVD Model Loaded Successfully!")

# # -----------------------------------------------------------
# # RECOMMENDATION FUNCTION WITH USER FILTERS
# # -----------------------------------------------------------
# def recommend_courses(user_id, n=5):
#     all_courses = df['course_id'].unique()
#     rated = df[df['user_id'] == user_id]['course_id'].unique()
#     remaining = [c for c in all_courses if c not in rated]

#     preds = [(c, model.predict(user_id, c).est) for c in remaining]
#     preds.sort(key=lambda x: x[1], reverse=True)

#     top_ids = [cid for cid, _ in preds]

#     recommended = df[df['course_id'].isin(top_ids)]

#     # APPLY USER FILTERS
#     recommended = recommended[recommended['difficulty_level'].isin(difficulty_choice)]

#     if cert_choice != "Both":
#         cert_flag = 1 if cert_choice == "Yes" else 0
#         recommended = recommended[recommended['certification_offered'] == cert_flag]

#     if material_choice != "Both":
#         mat_flag = 1 if material_choice == "Yes" else 0
#         recommended = recommended[recommended['study_material_available'] == mat_flag]

#     if "category" in df.columns:
#         recommended = recommended[recommended['category'] == user_interest]

#     return recommended.drop_duplicates().head(n)

# # -----------------------------------------------------------
# # OUTPUT: RECOMMENDATIONS
# # -----------------------------------------------------------
# if st.sidebar.button("Get Recommendations"):
#     st.subheader("🎯 Your Personalized Recommendations")

#     recs = recommend_courses(selected_user, num_rec)

#     if recs.empty:
#         st.warning("❌ No courses match your selected filters. Try different preferences.")
#     else:
#         for _, row in recs.iterrows():
#             stars = "⭐" * int(round(row['rating'] * 5))
#             cert = "✔ Yes" if row['certification_offered'] == 1 else "❌ No"
#             mat = "✔ Yes" if row['study_material_available'] == 1 else "❌ No"
#             img = f"https://source.unsplash.com/600x300/?course,{random.randint(1,100)}"

#             st.markdown(
#                 f"""
#                 <div class="card">
#                     <img src="{img}" width="100%" style="border-radius:10px;">
#                     <h4 class="course-title">{row['course_name']}</h4>
#                     <p class="instructor">👨‍🏫 {row['instructor']}</p>
#                     <p class="difficulty">📘 Difficulty: {row['difficulty_level']}</p>
#                     <p class="star-rating">{stars}</p>
#                     <p><b>Certification:</b> {cert}</p>
#                     <p><b>Study Material:</b> {mat}</p>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )






# import streamlit as st
# import pandas as pd
# import random
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------------------------------------
# # PAGE CONFIG
# # -----------------------------------------------------------
# st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# # -----------------------------------------------------------
# # CUSTOM CSS
# # -----------------------------------------------------------
# st.markdown("""
# <style>

# .card {
#     background: white;
#     padding: 20px;
#     border-radius: 12px;
#     margin-bottom: 22px;
#     box-shadow: 0 4px 12px rgba(0,0,0,0.10);
#     border-left: 6px solid #4c83ff;
# }

# .top-header {
#     font-size: 48px;
#     font-weight: 850;
#     color: #1b263b;
#     text-align: center;
# }

# .section-title {
#     font-size: 24px;
#     font-weight: 700;
#     color: #102542;
# }

# .course-title {
#     font-size: 22px;
#     font-weight: 750;
# }

# .star-rating {
#     font-size: 18px;
#     color: #f4c430;
# }

# </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------------------------
# # LOAD DATA
# # -----------------------------------------------------------
# @st.cache_data
# def load_data():
#     df = pd.read_csv("online_course_recommendation_v2.csv")

#     df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0})
#     df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0})
#     df['difficulty_level_encoded'] = df['difficulty_level'].map(
#         {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3})

#     return df

# df = load_data()

# # AUTO-DETECT CATEGORY COLUMN OR GENERATE ONE
# possible_cols = ["category", "course_domain", "subject"]
# detected = None
# for col in possible_cols:
#     if col in df.columns:
#         detected = col
#         break

# if not detected:
#     df["generated_category"] = df["course_name"].apply(lambda x: x.split()[0])
#     detected = "generated_category"

# categories = sorted(df[detected].unique())

# # -----------------------------------------------------------
# # TITLE
# # -----------------------------------------------------------
# st.markdown("<h1 class='top-header'>🎓 Course Recommendation System</h1>", unsafe_allow_html=True)
# st.write("AI-powered smart recommendations based on **your interests and preferences**.")
# st.write("---")

# # -----------------------------------------------------------
# # INPUT FIELDS (MAIN PAGE)
# # -----------------------------------------------------------
# st.markdown("<p class='section-title'>👉 Enter Your Preferences</p>", unsafe_allow_html=True)

# col1, col2 = st.columns(2)

# with col1:
#     user_id = st.selectbox("User ID:", sorted(df['user_id'].unique()))
#     difficulty_choice = st.selectbox("Difficulty Level:", df['difficulty_level'].unique())

# with col2:
#     category_choice = st.selectbox("Select Category / Interest:", categories)
#     certification_choice = st.radio("Certification Required?", ["Both", "Yes", "No"], index=0)

# study_choice = st.radio("Study Material Required?", ["Both", "Yes", "No"], index=0)

# st.write("---")

# # -----------------------------------------------------------
# # FILTER + SORT LOGIC (OUTPUT)
# # -----------------------------------------------------------
# def get_filtered_recommendations():
#     recs = df.copy()

#     recs = recs[recs[detected] == category_choice]
#     recs = recs[recs['difficulty_level'] == difficulty_choice]

#     if certification_choice != "Both":
#         cert_flag = 1 if certification_choice == "Yes" else 0
#         recs = recs[recs["certification_offered"] == cert_flag]

#     if study_choice != "Both":
#         mat_flag = 1 if study_choice == "Yes" else 0
#         recs = recs[recs["study_material_available"] == mat_flag]

#     # SORT BY RATING (highest first)
#     recs = recs.sort_values(by="rating", ascending=False)

#     return recs.drop_duplicates()

# # -----------------------------------------------------------
# # SHOW RECOMMENDATIONS
# # -----------------------------------------------------------
# if st.button("Get Recommendations"):
#     st.markdown("<p class='section-title'>🎯 Your Personalized Recommendations</p>", unsafe_allow_html=True)

#     results = get_filtered_recommendations()

#     if results.empty:
#         st.error("❌ No courses match your selected filters. Try different preferences.")
#     else:
#         for _, row in results.iterrows():
#             stars = "⭐" * int(round(row["rating"] * 5))
#             cert = "✔ Yes" if row["certification_offered"] == 1 else "❌ No"
#             mat = "✔ Yes" if row["study_material_available"] == 1 else "❌ No"
#             img = f"https://source.unsplash.com/600x300/?education,{random.randint(1,100)}"

#             st.markdown(f"""
#             <div class='card'>
#                 <img src="{img}" width="100%" style="border-radius:10px;">
#                 <h3 class="course-title">{row['course_name']}</h3>
#                 <p>👨‍🏫 <b>{row['instructor']}</b></p>
#                 <p>📘 Difficulty: {row['difficulty_level']}</p>
#                 <p class="star-rating">{stars}</p>
#                 <p><b>Certification:</b> {cert}</p>
#                 <p><b>Study Material:</b> {mat}</p>
#             </div>
#             """, unsafe_allow_html=True)

# st.write("---")





# import streamlit as st
# import pandas as pd
# import random
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity

# # -----------------------------------------------------------
# # PAGE CONFIG
# # -----------------------------------------------------------
# st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# # -----------------------------------------------------------
# # CUSTOM CSS FOR BEAUTIFUL UI
# # -----------------------------------------------------------
# st.markdown("""
# <style>
# body {
#     background-color: #f4f6fa;
# }

# /* MAIN HEADER */
# .main-header {
#     font-size: 48px;
#     font-weight: 900;
#     text-align: center;
#     margin-top: 10px;
#     margin-bottom: 5px;
#     color: #1b263b;
# }
# .sub-text {
#     font-size: 18px;
#     text-align: center;
#     color: #4a4a4a;
#     margin-bottom: 30px;
# }

# /* SIDEBAR TITLE */
# .sidebar-title {
#     font-size: 22px;
#     font-weight: 700;
#     color: #1b263b;
#     padding-bottom: 5px;
#     border-bottom: 2px solid #4c83ff;
# }

# /* GREEN RECOMMEND BUTTON */
# .stButton > button {
#     background-color: #22c55e;
#     color: white;
#     border-radius: 999px;
#     border: none;
#     padding: 0.6rem 1.4rem;
#     font-weight: 700;
#     font-size: 16px;
#     box-shadow: 0 4px 10px rgba(0,0,0,0.12);
# }
# .stButton > button:hover {
#     background-color: #16a34a;
#     box-shadow: 0 6px 16px rgba(0,0,0,0.18);
# }

# /* COURSE CARD */
# .card {
#     background: white;
#     padding: 22px;
#     border-radius: 18px;
#     margin-bottom: 24px;
#     box-shadow: 0 8px 20px rgba(15,23,42,0.10);
#     border-left: 6px solid #4c83ff;
# }
# .card img {
#     border-radius: 12px;
#     margin-bottom: 14px;
# }
# .course-title {
#     font-size: 22px;
#     font-weight: 800;
#     margin-bottom: 6px;
#     color: #111827;
# }
# .instructor {
#     font-size: 16px;
#     font-weight: 600;
#     color: #2563eb;
#     margin-bottom: 4px;
# }
# .label {
#     font-weight: 600;
#     color: #374151;
#     margin-bottom: 2px;
# }
# .star-rating {
#     font-size: 18px;
#     color: #facc15;
#     margin-bottom: 8px;
# }
# .badge {
#     display: inline-block;
#     padding: 4px 10px;
#     border-radius: 999px;
#     font-size: 12px;
#     font-weight: 600;
#     background-color: #eff6ff;
#     color: #1d4ed8;
#     margin-right: 6px;
# }
# </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------------------------
# # LOAD DATA
# # -----------------------------------------------------------
# @st.cache_data
# def load_data():
#     df = pd.read_csv("online_course_recommendation_v2.csv")

#     df["certification_offered"] = df["certification_offered"].map({"Yes": 1, "No": 0})
#     df["study_material_available"] = df["study_material_available"].map({"Yes": 1, "No": 0})
#     df["difficulty_level_encoded"] = df["difficulty_level"].map(
#         {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
#     )
#     return df

# df = load_data()

# # --------- CATEGORY COLUMN (agar naam alag ho to auto detect) ----------
# possible_cols = ["category", "course_domain", "subject"]
# detected = None
# for col in possible_cols:
#     if col in df.columns:
#         detected = col
#         break

# if not detected:
#     df["generated_category"] = df["course_name"].apply(lambda x: x.split()[0])
#     detected = "generated_category"

# categories = sorted(df[detected].unique())

# # -----------------------------------------------------------
# # HEADER
# # -----------------------------------------------------------
# st.markdown(
#     "<h1 class='main-header'>🎓 Course Recommendation System</h1>",
#     unsafe_allow_html=True,
# )
# st.markdown(
#     "<p class='sub-text'>AI-powered recommendations based on your interests and preferred learning style.</p>",
#     unsafe_allow_html=True,
# )
# st.write("---")

# # -----------------------------------------------------------
# # SIDEBAR – USER PREFERENCES
# # -----------------------------------------------------------
# st.sidebar.markdown(
#     "<p class='sidebar-title'>⚙️ Enter Your Preferences</p>",
#     unsafe_allow_html=True,
# )

# user_id = st.sidebar.selectbox("User ID:", sorted(df["user_id"].unique()))
# category_choice = st.sidebar.selectbox("Select Category / Interest:", categories)
# difficulty_choice = st.sidebar.selectbox(
#     "Difficulty Level:", sorted(df["difficulty_level"].unique())
# )
# certification_choice = st.sidebar.radio(
#     "Certification Required?", ["Both", "Yes", "No"], index=0
# )
# study_choice = st.sidebar.radio(
#     "Study Material Required?", ["Both", "Yes", "No"], index=0
# )
# num_rec = st.sidebar.slider("Number of Recommendations:", 1, 15, 6)

# get_recs = st.sidebar.button("✨ Get Recommendations")

# # -----------------------------------------------------------
# # SIMPLE FILTER-BASED RECOMMENDER (according to inputs)
# # -----------------------------------------------------------
# def get_filtered_recommendations():
#     recs = df.copy()

#     # category / interest
#     recs = recs[recs[detected] == category_choice]

#     # difficulty
#     recs = recs[recs["difficulty_level"] == difficulty_choice]

#     # certification filter
#     if certification_choice != "Both":
#         flag = 1 if certification_choice == "Yes" else 0
#         recs = recs[recs["certification_offered"] == flag]

#     # study material filter
#     if study_choice != "Both":
#         flag2 = 1 if study_choice == "Yes" else 0
#         recs = recs[recs["study_material_available"] == flag2]

#     # top courses by rating
#     recs = recs.sort_values(by="rating", ascending=False)

#     # only top N
#     return recs.drop_duplicates().head(num_rec)

# # -----------------------------------------------------------
# # MAIN OUTPUT – RECOMMENDATIONS
# # -----------------------------------------------------------
# if get_recs:
#     st.markdown("## 🎯 Your Personalized Recommendations")
#     results = get_filtered_recommendations()

#     if results.empty:
#         st.warning("❌ No courses found. Try changing your filters.")
#     else:
#         st.info(
#             f"Showing **{len(results)}** course(s) for User **{user_id}** "
#             f"in **{category_choice}** with difficulty **{difficulty_choice}**."
#         )

#         for _, row in results.iterrows():
#             stars = "⭐" * int(round(row["rating"] * 5))
#             cert = "✔ Yes" if row["certification_offered"] == 1 else "✖ No"
#             mat = "✔ Yes" if row["study_material_available"] == 1 else "✖ No"
#             img = f"https://source.unsplash.com/800x350/?education,{random.randint(1,100)}"

#             html = f"""
#             <div class="card">
#                 <img src="{img}" width="100%">
#                 <div class="badge">ID: {row['course_id']}</div>
#                 <div class="badge">{row[detected]}</div>
#                 <h3 class="course-title">{row['course_name']}</h3>
#                 <p class="instructor">👨‍🏫 {row['instructor']}</p>

#                 <p class="label">📘 Difficulty: {row['difficulty_level']}</p>
#                 <p class="star-rating">{stars}</p>

#                 <p class="label"><b>Certification:</b> {cert}</p>
#                 <p class="label"><b>Study Material:</b> {mat}</p>
#             </div>
#             """
#             st.markdown(html, unsafe_allow_html=True)
# else:
#     st.markdown("### 👈 Choose your preferences in the left panel and click **Get Recommendations**.")



# import streamlit as st
# import pandas as pd
# import random

# # -----------------------------------------------------------
# # PAGE CONFIG
# # -----------------------------------------------------------
# st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# # -----------------------------------------------------------
# # CUSTOM CSS
# # -----------------------------------------------------------
# st.markdown("""
# <style>

# body { background-color: #f4f6fa; }

# .main-header {
#     font-size: 48px;
#     font-weight: 900;
#     text-align: center;
#     margin-top: 10px;
#     color: #1b263b;
# }
# .sub-text {
#     font-size: 18px;
#     text-align: center;
#     color: #4a4a4a;
#     margin-bottom: 30px;
# }

# .sidebar-title {
#     font-size: 22px;
#     font-weight: 700;
#     color: #1b263b;
#     padding-bottom: 5px;
#     border-bottom: 2px solid #4c83ff;
# }

# /* Green Button */
# .stButton > button {
#     background-color: #22c55e;
#     color: white;
#     border-radius: 999px;
#     border: none;
#     padding: 0.6rem 1.4rem;
#     font-weight: 700;
#     font-size: 16px;
#     box-shadow: 0 4px 10px rgba(0,0,0,0.12);
# }
# .stButton > button:hover {
#     background-color: #16a34a;
#     box-shadow: 0 6px 16px rgba(0,0,0,0.18);
# }

# /* Course Card */
# .card {
#     background: white;
#     padding: 22px;
#     border-radius: 18px;
#     margin-bottom: 24px;
#     box-shadow: 0 8px 20px rgba(15,23,42,0.10);
#     border-left: 6px solid #4c83ff;
# }
# .card img {
#     border-radius: 12px;
#     margin-bottom: 14px;
# }
# .course-title {
#     font-size: 22px;
#     font-weight: 800;
#     color: #111827;
# }
# .instructor {
#     font-size: 16px;
#     font-weight: 600;
#     color: #2563eb;
# }
# .label {
#     font-weight: 600;
#     color: #374151;
# }
# .star-rating {
#     font-size: 18px;
#     color: #facc15;
# }

# </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------------------------
# # LOAD DATA
# # -----------------------------------------------------------
# @st.cache_data
# def load_data():
#     df = pd.read_csv("online_course_recommendation_v2.csv")

#     df["certification_offered"] = df["certification_offered"].map({"Yes": 1, "No": 0})
#     df["study_material_available"] = df["study_material_available"].map({"Yes": 1, "No": 0})

#     df["difficulty_level_encoded"] = df["difficulty_level"].map(
#         {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
#     )
#     return df

# df = load_data()

# # -----------------------------------------------------------
# # CATEGORY DETECTION OR AUTO-GENERATION
# # -----------------------------------------------------------
# possible_cols = ["category", "course_domain", "subject"]
# detected = None

# for col in possible_cols:
#     if col in df.columns:
#         detected = col
#         break

# # If no category column exists → generate one
# if not detected:
#     df["generated_category"] = df["course_name"].apply(lambda x: x.split()[0])
#     detected = "generated_category"

# categories = sorted(df[detected].unique())
# courses = sorted(df["course_name"].unique())

# # -----------------------------------------------------------
# # HEADER
# # -----------------------------------------------------------
# st.markdown("<h1 class='main-header'>🎓 Course Recommendation System</h1>", unsafe_allow_html=True)
# st.markdown("<p class='sub-text'>AI-powered smart recommendations based on your interests and preferred learning style.</p>", unsafe_allow_html=True)

# st.write("---")

# # -----------------------------------------------------------
# # SIDEBAR INPUTS
# # -----------------------------------------------------------
# st.sidebar.markdown("<p class='sidebar-title'>⚙️ Enter Your Preferences</p>", unsafe_allow_html=True)

# user_id = st.sidebar.selectbox("User ID:", sorted(df["user_id"].unique()))

# selected_course = st.sidebar.selectbox("Select a Course:", courses)

# # Auto-select category based on selected course
# category_choice = df.loc[df["course_name"] == selected_course, detected].values[0]

# difficulty_choice = st.sidebar.selectbox(
#     "Preferred Difficulty:", sorted(df["difficulty_level"].unique())
# )

# certification_choice = st.sidebar.radio("Certification Required?", ["Both", "Yes", "No"], index=0)
# study_choice = st.sidebar.radio("Study Material Required?", ["Both", "Yes", "No"], index=0)

# num_rec = st.sidebar.slider("Number of Recommendations:", 1, 15, 6)

# button = st.sidebar.button("✨ Get Recommendations")

# # -----------------------------------------------------------
# # FILTER LOGIC
# # -----------------------------------------------------------
# def get_recommendations():
#     recs = df.copy()

#     # filter by selected category
#     recs = recs[recs[detected] == category_choice]

#     # filter by difficulty
#     recs = recs[recs["difficulty_level"] == difficulty_choice]

#     # certification filter
#     if certification_choice != "Both":
#         recs = recs[recs["certification_offered"] == (1 if certification_choice == "Yes" else 0)]

#     # study material filter
#     if study_choice != "Both":
#         recs = recs[recs["study_material_available"] == (1 if study_choice == "Yes" else 0)]

#     # ⭐ sort always by highest rating
#     recs = recs.sort_values(by="rating", ascending=False)

#     return recs.drop_duplicates().head(num_rec)

# # -----------------------------------------------------------
# # SHOW RECOMMENDATIONS
# # -----------------------------------------------------------
# if button:

#     st.markdown(f"## 🎯 Showing Top **{num_rec}** Recommendations for You")
#     st.write(
#         f"Based on course **{selected_course}**, category **{category_choice}**, "
#         f"difficulty **{difficulty_choice}**, certification **{certification_choice}**, "
#         f"study material **{study_choice}**."
#     )
#     st.write("---")

#     results = get_recommendations()

#     if results.empty:
#         st.warning("❌ No matching courses found. Try different settings.")
#     else:
#         for _, row in results.iterrows():
#             stars = "⭐" * int(round(row["rating"] * 5))
#             rating_numeric = round(row["rating"], 2)
#             cert = "✔ Yes" if row["certification_offered"] == 1 else "✖ No"
#             mat = "✔ Yes" if row["study_material_available"] == 1 else "✖ No"
#             img = f"https://source.unsplash.com/800x350/?education,{random.randint(1,100)}"

#             card_html = f"""
#             <div class="card">
#                 <img src="{img}" width="100%">
#                 <h3 class="course-title">{row['course_name']}</h3>
#                 <p class="instructor">👨‍🏫 {row['instructor']}</p>

#                 <p class="label">📘 Difficulty: {row['difficulty_level']}</p>

#                 <p class="star-rating">{stars}</p>
#                 <p class="label"><b>Rating:</b> {rating_numeric} / 5.0</p>

#                 <p class="label"><b>Certification:</b> {cert}</p>
#                 <p class="label"><b>Study Material:</b> {mat}</p>
#             </div>
#             """

#             st.markdown(card_html, unsafe_allow_html=True)








# import streamlit as st
# import pandas as pd
# import random

# # -----------------------------------------------------------
# # PAGE CONFIG
# # -----------------------------------------------------------
# st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")


# # -----------------------------------------------------------
# # PREMIUM CUSTOM CSS
# # -----------------------------------------------------------
# st.markdown("""
# <style>

# @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

# * {
#     font-family: 'Poppins', sans-serif;
# }

# body { 
#     background-color: #f3f4f6;
# }

# /* HEADER */
# .main-header {
#     font-size: 50px;
#     font-weight: 800;
#     text-align: center;
#     margin-top: 10px;
#     color: #111827;
# }

# .sub-text {
#     font-size: 18px;
#     text-align: center;
#     color: #4b5563;
#     margin-bottom: 30px;
# }

# /* SIDEBAR TITLE */
# .sidebar-title {
#     font-size: 22px;
#     font-weight: 700;
#     color: #1f2937;
#     padding-bottom: 5px;
#     border-bottom: 2px solid #3b82f6;
# }

# /* MODERN GREEN BUTTON */
# .stButton > button {
#     background: linear-gradient(135deg, #22c55e, #16a34a);
#     color: white;
#     border-radius: 999px;
#     border: none;
#     padding: 0.7rem 1.6rem;
#     font-weight: 700;
#     font-size: 16px;
#     box-shadow: 0 4px 12px rgba(0,0,0,0.12);
# }
# .stButton > button:hover {
#     background: linear-gradient(135deg, #16a34a, #15803d);
#     box-shadow: 0 6px 18px rgba(0,0,0,0.18);
# }

# /* COURSE CARD */
# .card {
#     background: white;
#     padding: 25px;
#     border-radius: 20px;
#     margin-bottom: 28px;
#     box-shadow: 0 10px 25px rgba(0,0,0,0.10);
#     border-left: 6px solid #3b82f6;
#     transition: transform 0.2s ease-in-out;
# }
# .card:hover {
#     transform: translateY(-6px);
# }

# /* COURSE IMAGE */
# .card img {
#     border-radius: 14px;
#     margin-bottom: 14px;
# }

# /* COURSE TITLE */
# .course-title {
#     font-size: 24px;
#     font-weight: 800;
#     color: #111827;
#     margin-bottom: -5px;
# }

# /* Instructor */
# .instructor {
#     font-size: 16px;
#     font-weight: 600;
#     color: #2563eb;
# }

# /* Labels */
# .label {
#     font-weight: 600;
#     color: #374151;
# }

# /* STAR RATING */
# .star-rating {
#     font-size: 20px;
#     letter-spacing: 2px;
#     color: #fbbf24;
# }

# </style>
# """, unsafe_allow_html=True)


# # -----------------------------------------------------------
# # LOAD DATA
# # -----------------------------------------------------------
# @st.cache_data
# def load_data():
#     df = pd.read_csv("online_course_recommendation_v2.csv")

#     # Convert Yes/No to 1/0
#     df["certification_offered"] = df["certification_offered"].map({"Yes": 1, "No": 0})
#     df["study_material_available"] = df["study_material_available"].map({"Yes": 1, "No": 0})

#     # Difficulty encoding
#     df["difficulty_level_encoded"] = df["difficulty_level"].map(
#         {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
#     )
#     return df


# df = load_data()


# # -----------------------------------------------------------
# # CATEGORY DETECTION OR AUTO-CREATION
# # -----------------------------------------------------------
# possible_cols = ["category", "course_domain", "subject"]
# detected = None

# for col in possible_cols:
#     if col in df.columns:
#         detected = col
#         break

# if not detected:
#     df["generated_category"] = df["course_name"].apply(lambda x: x.split()[0])
#     detected = "generated_category"

# categories = sorted(df[detected].unique())
# courses = sorted(df["course_name"].unique())


# # -----------------------------------------------------------
# # HEADER UI
# # -----------------------------------------------------------
# st.markdown("<h1 class='main-header'>🎓 Course Recommendation System</h1>", unsafe_allow_html=True)
# st.markdown("<p class='sub-text'>AI-powered recommendations crafted from your interests and learning preferences.</p>", unsafe_allow_html=True)

# st.write("---")


# # -----------------------------------------------------------
# # SIDEBAR INPUT PANEL
# # -----------------------------------------------------------
# st.sidebar.markdown("<p class='sidebar-title'>⚙️ Enter Your Preferences</p>", unsafe_allow_html=True)

# user_id = st.sidebar.selectbox("User ID:", sorted(df["user_id"].unique()))

# selected_course = st.sidebar.selectbox("Choose a Course:", courses)

# # Auto-detect category based on selected course
# category_choice = df.loc[df["course_name"] == selected_course, detected].values[0]

# difficulty_choice = st.sidebar.selectbox("Preferred Difficulty:", sorted(df["difficulty_level"].unique()))

# certification_choice = st.sidebar.radio("Certification Required?", ["Both", "Yes", "No"], index=0)
# study_choice = st.sidebar.radio("Study Material Required?", ["Both", "Yes", "No"], index=0)

# num_rec = st.sidebar.slider("Number of Recommendations:", 1, 15, 6)

# button = st.sidebar.button("✨ Get Recommendations")


# # -----------------------------------------------------------
# # FILTER LOGIC
# # -----------------------------------------------------------
# def get_recommendations():
#     recs = df.copy()

#     recs = recs[recs[detected] == category_choice]
#     recs = recs[recs["difficulty_level"] == difficulty_choice]

#     if certification_choice != "Both":
#         recs = recs[recs["certification_offered"] == (1 if certification_choice == "Yes" else 0)]

#     if study_choice != "Both":
#         recs = recs[recs["study_material_available"] == (1 if study_choice == "Yes" else 0)]

#     # sort by highest rating
#     recs = recs.sort_values(by="rating", ascending=False)

#     return recs.drop_duplicates().head(num_rec)


# # -----------------------------------------------------------
# # SHOW RESULTS
# # -----------------------------------------------------------
# if button:

#     st.markdown(f"## 🎯 Top {num_rec} Personalized Recommendations")
#     st.write(
#         f"Based on **{selected_course}**, category **{category_choice}**, "
#         f"difficulty **{difficulty_choice}**, certification **{certification_choice}**, "
#         f"study material **{study_choice}**."
#     )
#     st.write("---")

#     results = get_recommendations()

#     if results.empty:
#         st.warning("❌ No matching courses found. Try adjusting your filters.")

#     else:
#         for _, row in results.iterrows():
#             stars = "⭐" * int(round(row["rating"] * 5))
#             rating_numeric = round(row["rating"], 2)
#             cert = "✔ Yes" if row["certification_offered"] == 1 else "✖ No"
#             mat = "✔ Yes" if row["study_material_available"] == 1 else "✖ No"

#             img = f"https://source.unsplash.com/800x350/?learning,education,{random.randint(1,100)}"

#             card_html = f"""
# <div class="card">
#     <img src="{img}" width="100%">
#     <h3 class="course-title">{row['course_name']}</h3>

#     <p class="instructor">👩‍🏫 {row['instructor']}</p>
    
#     <p class="label">📘 Difficulty: {row['difficulty_level']}</p>
    
#     <p class="star-rating">{stars}</p>
#     <p class="label"><b>Rating:</b> {rating_numeric} / 5.0</p>

#     <p class="label"><b>Certification:</b> {cert}</p>
#     <p class="label"><b>Study Material:</b> {mat}</p>
# </div>
# """
            




#             st.markdown(card_html, unsafe_allow_html=True)



import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import random

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

# -----------------------------------------------------------
# PREMIUM CUSTOM CSS
# -----------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

* {
    font-family: sans-serif;
}

body { background-color: #f3f4f6; }

/* HEADER */
.main-header {
    font-size: 50px;
    font-weight: 800;
    text-align: center;
    margin-top: 10px;
    color: #111827;
}

.sub-text {
    font-size: 18px;
    text-align: center;
    color: #4b5563;
    margin-bottom: 30px;
}

/* SIDEBAR TITLE */
.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: #1f2937;
    padding-bottom: 5px;
    border-bottom: 2px solid #3b82f6;
}

/* MODERN GREEN BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    border-radius: 999px;
    border: none;
    padding: 0.7rem 1.6rem;
    font-weight: 700;
    font-size: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #16a34a, #15803d);
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("online_course_recommendation_v2.csv")

    df["certification_offered"] = df["certification_offered"].map({"Yes": 1, "No": 0})
    df["study_material_available"] = df["study_material_available"].map({"Yes": 1, "No": 0})

    df["difficulty_level_encoded"] = df["difficulty_level"].map(
        {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    )
    return df

df = load_data()


# -----------------------------------------------------------
# CATEGORY DETECTION
# -----------------------------------------------------------
possible_cols = ["category", "course_domain", "subject"]
detected = None

for col in possible_cols:
    if col in df.columns:
        detected = col
        break

if not detected:
    df["generated_category"] = df["course_name"].apply(lambda x: x.split()[0])
    detected = "generated_category"

categories = sorted(df[detected].unique())
courses = sorted(df["course_name"].unique())


# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown("<h1 class='main-header'>🎓 Course Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>AI-powered recommendations tailored to your learning needs.</p>", unsafe_allow_html=True)
st.write("---")


# -----------------------------------------------------------
# SIDEBAR INPUT PANEL
# -----------------------------------------------------------
st.sidebar.markdown("<p class='sidebar-title'>⚙️ Enter Your Preferences</p>", unsafe_allow_html=True)

user_id = st.sidebar.selectbox("User ID:", sorted(df["user_id"].unique()))
selected_course = st.sidebar.selectbox("Choose a Course:", courses)

category_choice = df.loc[df["course_name"] == selected_course, detected].values[0]
difficulty_choice = st.sidebar.selectbox("Preferred Difficulty:", sorted(df["difficulty_level"].unique()))

certification_choice = st.sidebar.radio("Certification Required?", ["Both", "Yes", "No"], index=0)
study_choice = st.sidebar.radio("Study Material Required?", ["Both", "Yes", "No"], index=0)

num_rec = st.sidebar.slider("Number of Recommendations:", 1, 20, 6)

button = st.sidebar.button("✨ Get Recommendations")


# -----------------------------------------------------------
# FILTER FUNCTION
# -----------------------------------------------------------
def get_recommendations():
    recs = df.copy()

    recs = recs[recs[detected] == category_choice]
    recs = recs[recs["difficulty_level"] == difficulty_choice]

    if certification_choice != "Both":
        recs = recs[recs["certification_offered"] == (1 if certification_choice == "Yes" else 0)]

    if study_choice != "Both":
        recs = recs[recs["study_material_available"] == (1 if study_choice == "Yes" else 0)]

    recs = recs.sort_values(by="rating", ascending=False)

    return recs.drop_duplicates().head(num_rec)


# -----------------------------------------------------------
# SHOW RESULTS
# -----------------------------------------------------------
if button:

    st.markdown(f"## 🎯 Top {num_rec} Personalized Recommendations")
    st.write(
        f"Based on **{selected_course}**, category **{category_choice}**, "
        f"difficulty **{difficulty_choice}**, certification **{certification_choice}**, "
        f"study material **{study_choice}**."
    )
    st.write("---")

    results = get_recommendations()

    if results.empty:
        st.warning("❌ No matching courses found. Try different settings.")

    else:
        for _, row in results.iterrows():

            # Generate dynamic image
            img = f"https://source.unsplash.com/800x350/?education,{random.randint(1,100)}"

            rating_numeric = round(row["rating"], 2)
            cert = "✔ Yes" if row["certification_offered"] == 1 else "❌ No"
            mat = "✔ Yes" if row["study_material_available"] == 1 else "❌ No"

            # -------------------------------------------------------
            # PREMIUM HTML CARD (NO HTML SHOWING AS TEXT)
            # -------------------------------------------------------
            card_html = f"""
            <div style="
                width: 100%;
                background: #ffffff;
                padding: 24px;
                border-radius: 20px;
                margin: 25px 0;
                 border-left: 8px solid #3B82F6;
                box-shadow: 0px 6px 18px rgba(59,130,246,0.15);
                font-family: 'Inter', sans-serif;            "
            onmouseover="this.style.transform='translateY(-6px)'; 
             this.style.boxShadow='0px 10px 25px rgba(0,0,0,0.18)'"
onmouseout="this.style.transform='translateY(0px)'; 
            this.style.boxShadow='0px 4px 14px rgba(0,0,0,0.08)'">
                <div style="
    width: 100%;
    # text-align: center;
    font-size: 20px;
    margin-bottom: 12px;
    color: #3B82F6;
">
     📖
</div>

                <h2 style="font-size: 26px; font-weight: 790; color: #111827;">
                    {row['course_name']}
                </h2>

                <p style="font-size: 17px; font-weight: 600; color: #2563eb;">
                    👨‍🏫 {row['instructor']}
                </p>

                <span style="
                    background: #dbeafe;
                    color: #1e40af;
                    padding: 6px 14px;
                    border-radius: 30px;
                    font-size: 14px;
                    font-weight: 600;
                    margin-right: 8px;
                ">📘 {row['difficulty_level']}</span>
            
                <span style="
                    background: #fef3c7;
                    color: #b45309;
                    padding: 1px 5px;
                    border-radius: 30px;
                    font-size: 14px;
                    font-weight: 600;
                ">⭐ {rating_numeric} / 5.0</span>

                <br><br>

                <p style="font-size: 16px; font-weight: 600; color: #374151;">
                    🎓 Certification: {cert}
                </p>
                <p style="font-size: 16px; font-weight: 600; color: #374151;">
                    📚 Study Material: {mat}
                </p>

            </div>
            """

            components.html(card_html, height=350)