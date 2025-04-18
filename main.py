import numpy as np
import streamlit as st
import pandas as pd
from scipy import sparse
import random

class BaseRecommandation:
    # Load and prepare data (titles & similarity matrices)
    titles = pd.read_pickle('database/sample_title.pkl').reset_index(drop=True)  # Ensure index matches matrix
    cosine_sim = sparse.load_npz("database/cosine_similarity_matrix.npz")
    dot_sim = sparse.load_npz("database/sparse_dot_matrix.npz")

    # FIX: Load jaccard and unwrap if needed
    jaccard_sim_loaded = np.load("database/jaccard_similarity_matrix.npz", allow_pickle=True)
    jaccard_sim = jaccard_sim_loaded['arr_0']
    if isinstance(jaccard_sim, np.ndarray) and jaccard_sim.shape == (1,):
        jaccard_sim = jaccard_sim[0]
    if isinstance(jaccard_sim, np.matrix):
        jaccard_sim = np.asarray(jaccard_sim)

    # Init
    def __init__(self, movie_input):
        self.movie_input = movie_input

    def get_index(self):
        try:
            idx = BaseRecommandation.titles[BaseRecommandation.titles == self.movie_input].index[0]
            return idx
        except IndexError:
            return None

    @staticmethod
    def quality_check(recommendations, scores, threshold, method):
        st.subheader("📽️ Top Recommendations:")
        if len(scores) == 0:
            st.error("😕 No recommendations found. Try another movie title.")
            return
        if all(s < threshold for s in scores):
            st.info("🎭 These may not be blockbusters... but every movie deserves a shot!")

            cosine_msgs = [
                "😅 That wasn't a hit. Maybe try a different movie title?",
                "🤖 Cosine similarity flopped. Try Jaccard next?",
                "💡 Tip: Cosine works better with rich metadata. Maybe Weighted will perform better.",
            ]
            jaccard_msgs = [
                "🎲 Genre-based similarity is tricky. Try something more personalized?",
                "🎯 Maybe the genres just didn’t line up. Try cosine or weighted?",
            ]
            weighted_msgs = [
                "🧐 Hmm... Nothing impressive here. Try tweaking the weights!",
                "💡 Tip: Some combos just don't vibe. Try increasing the cosine weight?",
            ]

            if method == "Cosine Similarity":
                st.warning(random.choice(cosine_msgs))
            elif method == "Jaccard Genre based Recommandation":
                st.warning(random.choice(jaccard_msgs))
            elif method == "Weighted Recommendation":
                st.warning(random.choice(weighted_msgs))

        # Always show recommendations
        for i, (index, row) in enumerate(recommendations.iterrows()):
            s = scores[i]
            if s >= threshold + 0.2:
                emoji = "🌟🌟🌟 Excellent"
            elif s >= threshold + 0.1:
                emoji = "🌟🌟 Good"
            elif s >= threshold:
                emoji = "🌟 Fair"
            else:
                emoji = "💤 Weak"
            st.markdown(f"**{i + 1}. {row['title']}** — Score: `{s:.4f}` {emoji}")
            st.progress(min(s, 1.0))

class cosine_recommandation(BaseRecommandation):
    def __init__(self, movie_input):
        super().__init__(movie_input)

    def get_recommandation(self):
        index = self.get_index()
        if index is None:
            return [], []
        else:
            cosine_score = self.cosine_sim[index].toarray().flatten()
            cosine_score[index] = -1
            top_indices = np.argsort(cosine_score)[::-1][:5]
            recommendations = pd.DataFrame({'title': self.titles.iloc[top_indices]})
            return recommendations, cosine_score[top_indices]


class jaccardRecommandation(BaseRecommandation):
    def __init__(self, movie_input):
        super().__init__(movie_input)

    def getRecommandation(self):
        index = self.get_index()
        if index is None:
            return [], []
        else:
            jaccard_score = self.jaccard_sim[index]
            jaccard_score[index] = -1
            top_indices = np.argsort(jaccard_score)[::-1][:5]
            recommendations = pd.DataFrame({'title': self.titles.iloc[top_indices]})
            return recommendations, jaccard_score[top_indices]


class WeightedRecommendation(BaseRecommandation):
    def __init__(self, cosine_weight, jaccard_weight, dot_weight, movie_input):
        super().__init__(movie_input)
        self.cosine_weight = cosine_weight
        self.jaccard_weight = jaccard_weight
        self.dot_weight = dot_weight

    def get_recommendations(self):
        idx = self.get_index()
        if idx is None:
            return [], []

        cosine_score = BaseRecommandation.cosine_sim[idx].toarray().flatten()
        jaccard_score = BaseRecommandation.jaccard_sim[idx]
        dot_score = BaseRecommandation.dot_sim[idx].toarray().flatten()

        final_score = (
            self.cosine_weight * cosine_score +
            self.jaccard_weight * jaccard_score +
            self.dot_weight * dot_score
        )
        final_score[idx] = -1
        top_indices = np.argsort(final_score)[::-1][:5]
        recommendations = pd.DataFrame({'title': self.titles.iloc[top_indices]})
        return recommendations, final_score[top_indices]


# ------------------------ Streamlit Frontend ------------------------

st.set_page_config(page_title="Recommendation System", layout="wide")
st.title("🎩 Experiments with Recommendation")

st.sidebar.title("🔍 Choose Recommendation Method")
method = st.sidebar.selectbox("Select Algorithm", [
    "Cosine Similarity",
    "Weighted Recommendation",
    "Jaccard Genre based Recommandation"
])

title_df = BaseRecommandation.titles
movie_list = title_df.tolist()
selected_movie = st.selectbox("🎬 Choose a Movie", movie_list)

if method == "Weighted Recommendation":
    st.sidebar.subheader("⚖️ Set Weights")
    cosine_weight = st.sidebar.slider("Cosine Weight", 0.0, 1.0, 0.4, 0.05)
    jaccard_weight = st.sidebar.slider("Jaccard Weight", 0.0, 1.0, 0.3, 0.05)
    dot_weight = st.sidebar.slider("Dot Weight", 0.0, 1.0, 0.3, 0.05)
    if abs(cosine_weight + jaccard_weight + dot_weight - 1.0) > 0.01:
        st.sidebar.error("Weights must sum to 1. Adjust accordingly.")
    else:
        score_threshold = st.sidebar.slider("Minimum score to be considered good", 0.0, 1.0, 0.5, 0.05)
        generate_button = st.sidebar.button("🎯 Generate Recommendations")
        if generate_button:
            obj = WeightedRecommendation(cosine_weight, jaccard_weight, dot_weight, selected_movie)
            rec, score = obj.get_recommendations()
            BaseRecommandation.quality_check(rec, score, score_threshold,"Weighted Recommendation")

elif method == "Cosine Similarity":
    score_threshold = st.sidebar.slider("Minimum score to be considered good", 0.0, 1.0, 0.5, 0.05)
    generate_button = st.sidebar.button("🎯 Generate Recommendations")
    if generate_button:
        recommender = cosine_recommandation(selected_movie)
        recs, scores = recommender.get_recommandation()
        BaseRecommandation.quality_check(recs, scores, score_threshold,"Cosine Similarity")

elif method == "Jaccard Genre based Recommandation":
    score_threshold = st.sidebar.slider("Minimum score to be considered good", 0.0, 1.0, 0.5, 0.05)
    generate_button = st.sidebar.button("🎯 Generate Recommendations")
    if generate_button:
        recommander = jaccardRecommandation(selected_movie)
        rec, score = recommander.getRecommandation()
        BaseRecommandation.quality_check(rec, score, score_threshold,"Jaccard Genre based Recommandation")
