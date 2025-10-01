import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="نظام توصية الأفلام",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    .recommendation-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .movie-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .section-header {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """تحميل البيانات والنماذج المحفوظة"""
    
    possible_files = [
        'movie_data.pkl',
        'movie_recommendation_data.pkl', 
        'recommendation_data.pkl'
    ]
    
    for file_name in possible_files:
        try:
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
            st.success(f"✅ تم تحميل البيانات بنجاح من: {file_name}")
            return data
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"⚠️ خطأ في تحميل {file_name}: {e}")
            continue
    
    st.warning("📝 لم يتم العثور على ملف بيانات، جاري إنشاء بيانات تجريبية...")
    
    try:
        movies_data = {
            'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'title': [
                'The Dark Knight', 'Inception', 'Interstellar', 
                'The Matrix', 'Pulp Fiction', 'The Godfather',
                'Fight Club', 'Forrest Gump', 'The Shawshank Redemption',
                'The Avengers'
            ],
            'vote_average': [9.0, 8.8, 8.6, 8.7, 8.9, 9.2, 8.8, 8.8, 9.3, 8.0],
            'vote_count': [1000, 900, 800, 850, 950, 1200, 880, 920, 1300, 750],
            'genres': [
                ['Action', 'Crime', 'Drama'],
                ['Action', 'Sci-Fi', 'Thriller'],
                ['Adventure', 'Drama', 'Sci-Fi'],
                ['Action', 'Sci-Fi'],
                ['Crime', 'Drama'],
                ['Crime', 'Drama'],
                ['Drama'],
                ['Drama', 'Romance'],
                ['Drama'],
                ['Action', 'Adventure', 'Sci-Fi']
            ]
        }
        
        movies_df = pd.DataFrame(movies_data)
        
        np.random.seed(42)
        cosine_sim_content = np.random.rand(len(movies_df), len(movies_df))
        
        user_movie_matrix = pd.DataFrame(
            np.random.randint(0, 6, size=(50, len(movies_df))),
            columns=movies_df['movie_id'].values
        )
        
        ratings_data = []
        for user_id in range(50):
            for movie_id in movies_df['movie_id']:
                if user_movie_matrix.loc[user_id, movie_id] > 0:
                    ratings_data.append({
                        'user_id': user_id + 1,
                        'movie_id': movie_id,
                        'rating': user_movie_matrix.loc[user_id, movie_id]
                    })
        
        ratings_df = pd.DataFrame(ratings_data)
        
        data = {
            'movies': movies_df,
            'cosine_sim_content': cosine_sim_content,
            'user_movie_matrix': user_movie_matrix,
            'ratings_df': ratings_df
        }
        
        with open('movie_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        st.success("✅ تم إنشاء وحفظ بيانات تجريبية بنجاح!")
        return data
        
    except Exception as e:
        st.error(f"❌ فشل في إنشاء البيانات: {e}")
        return None

def get_content_recommendations(title, movies_df, cosine_sim, n_recommendations=10):
    """توصيات Based-Content"""
    try:
        idx = movies_df[movies_df['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        
        recommendations = movies_df.iloc[movie_indices][['title', 'vote_average', 'vote_count', 'genres']].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        return recommendations
    except Exception as e:
        st.error(f"❌ خطأ في التوصيات المبنية على المحتوى: {str(e)}")
        return None

def item_based_recommendations(movie_id, movies_df, user_movie_matrix, n_recommendations=10):
    """توصيات Item-Based Collaborative Filtering"""
    try:
        sparse_matrix = csr_matrix(user_movie_matrix.values)
        item_similarity = cosine_similarity(sparse_matrix.T)
        
        item_similarity_df = pd.DataFrame(
            item_similarity, 
            index=user_movie_matrix.columns, 
            columns=user_movie_matrix.columns
        )
        
        if movie_id not in item_similarity_df.columns:
            return None
            
        similar_scores = item_similarity_df[movie_id].sort_values(ascending=False)
        similar_movies = similar_scores.iloc[1:n_recommendations+1]
        
        recommendations = []
        for similar_movie_id, score in similar_movies.items():
            movie_info = movies_df[movies_df['movie_id'] == similar_movie_id]
            if not movie_info.empty:
                title = movie_info['title'].values[0]
                vote_avg = movie_info['vote_average'].values[0]
                genres = movie_info['genres'].values[0]
                recommendations.append({
                    'title': title,
                    'similarity_score': score,
                    'vote_average': vote_avg,
                    'genres': genres,
                    'movie_id': similar_movie_id
                })
        
        return pd.DataFrame(recommendations)
    except Exception as e:
        st.error(f"❌ خطأ في التوصيات المبنية على العناصر: {str(e)}")
        return None

def main():
    """التطبيق الرئيسي"""
    
    st.markdown('<h1 class="main-header">🎬 نظام توصية الأفلام المتقدم</h1>', unsafe_allow_html=True)

    with st.spinner("جاري تحميل البيانات..."):
        data = load_data()
    
    if data is None:
        st.error("❌ لا يمكن تشغيل التطبيق بدون بيانات")
        return
    
    movies_df = data['movies']
    cosine_sim_content = data['cosine_sim_content']
    user_movie_matrix = data['user_movie_matrix']
    ratings_df = data['ratings_df']
    
    st.sidebar.title("⚙️ إعدادات التوصية")
    
    recommendation_type = st.sidebar.selectbox(
        "اختر نوع التوصية:",
        ["توصيات مبنية على المحتوى", "توصيات مبنية على العناصر"]
    )
    
    n_recommendations = st.sidebar.slider("عدد التوصيات:", min_value=5, max_value=20, value=10)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 إحصائيات النظام")
    st.sidebar.metric("🎬 عدد الأفلام", len(movies_df))
    st.sidebar.metric("👥 عدد المستخدمين", len(user_movie_matrix))
    st.sidebar.metric("⭐ عدد التقييمات", len(ratings_df))
    st.sidebar.metric("📈 متوسط التقييم", f"{movies_df['vote_average'].mean():.2f}")
    
    if recommendation_type == "توصيات مبنية على المحتوى":
        st.markdown('<div class="section-header">', unsafe_allow_html=True)
        st.header("🎯 توصيات مبنية على المحتوى")
        st.write("هذا النظام يوصي بأفلام مشابهة بناءً على النوع، الممثلين، المخرج، والقصّة.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            movie_titles = movies_df['title'].sort_values().tolist()
            selected_movie = st.selectbox("اختر فيلم:", movie_titles)
        
        with col2:
            st.write("")
            st.write("")
            if st.button("🎯 احصل على التوصيات", use_container_width=True):
                with st.spinner("جاري البحث عن أفلام مشابهة..."):
                    recommendations = get_content_recommendations(
                        selected_movie, movies_df, cosine_sim_content, n_recommendations
                    )
                    
                if recommendations is not None and not recommendations.empty:
                    st.success(f"✅ تم العثور على {len(recommendations)} توصية لفيلم **{selected_movie}**")
                    
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <div class="movie-title">{row['title']}</div>
                                <div>🎭 <strong>الأنواع:</strong> {', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres']}</div>
                                <div>⭐ <strong>التقييم:</strong> <span class="metric-value">{row['vote_average']}</span></div>
                                <div>📊 <strong>عدد التقييمات:</strong> {row['vote_count']}</div>
                                <div>🔍 <strong>مستوى التشابه:</strong> <span class="metric-value">{row['similarity_score']:.3f}</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("❌ لم نتمكن من العثور على توصيات لهذا الفيلم.")
    
    elif recommendation_type == "توصيات مبنية على العناصر":
        st.markdown('<div class="section-header">', unsafe_allow_html=True)
        st.header("👥 توصيات مبنية على العناصر")
        st.write("هذا النظام يحلل أنماط التقييم بين الأفلام المختلفة لتقديم توصيات دقيقة.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            movie_titles = movies_df['title'].sort_values().tolist()
            selected_movie = st.selectbox("اختر فيلم:", movie_titles)
        
        with col2:
            st.write("")
            st.write("")
            if st.button("👥 احصل على التوصيات", use_container_width=True):
                selected_movie_id = movies_df[movies_df['title'] == selected_movie]['movie_id'].values[0]
                
                with st.spinner("جاري تحليل أنماط التقييم..."):
                    recommendations = item_based_recommendations(
                        selected_movie_id, movies_df, user_movie_matrix, n_recommendations
                    )
                    
                if recommendations is not None and not recommendations.empty:
                    st.success(f"✅ تم العثور على {len(recommendations)} توصية بناءً على فيلم **{selected_movie}**")
                    
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <div class="movie-title">{row['title']}</div>
                                <div>🎭 <strong>الأنواع:</strong> {', '.join(row['genres']) if isinstance(row['genres'], list) else row['genres']}</div>
                                <div>⭐ <strong>التقييم:</strong> <span class="metric-value">{row['vote_average']}</span></div>
                                <div>🔍 <strong>مستوى التشابه:</strong> <span class="metric-value">{row['similarity_score']:.3f}</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("❌ لم نتمكن من العثور على توصيات لهذا الفيلم.")
    
    st.markdown("---")
    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.header("🔍 استكشاف البيانات")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 أفضل 10 أفلام تقييماً")
        top_movies = movies_df.nlargest(10, 'vote_average')[['title', 'vote_average', 'vote_count']]
        st.dataframe(top_movies, use_container_width=True, height=400)
    
    with col2:
        st.subheader("📊 Distribution of rating ")
        fig, ax = plt.subplots(figsize=(8, 4))
        movies_df['vote_average'].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Avg rates')
        ax.set_ylabel('Movie count')
        ax.set_title('Distribution of movies rating')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

if __name__ == "__main__":
    main()