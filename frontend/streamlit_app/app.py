import streamlit as st

import api_client
import charts
import components

st.set_page_config(page_title="CINEIQ", layout="wide")

st.title("CINEIQ Movie Recommender")

user_id = st.sidebar.number_input("User ID", min_value=1, value=1, step=1)

with st.sidebar:
    top_k = st.slider("Top K", min_value=5, max_value=20, value=10)
    apply_sentiment = st.checkbox("Apply sentiment rerank", value=True)

recommend_tab, similar_tab, profile_tab, insights_tab = st.tabs(
    ["Recommendations", "Similar Movies", "Taste Profile", "Model Insights"]
)

with recommend_tab:
    st.subheader("Recommendations")
    if st.button("Get Recommendations"):
        payload = {"user_id": user_id, "top_k": top_k, "apply_sentiment": apply_sentiment}
        data = api_client.post("/recommend", payload)
        for rec in data.get("recommendations", []):
            components.recommendation_card(rec)

with similar_tab:
    st.subheader("Similar Movies")
    movie_id = st.number_input("Movie ID", min_value=1, value=1, step=1)
    if st.button("Find Similar"):
        data = api_client.post("/similar", {"movie_id": movie_id, "top_k": top_k})
        for item in data.get("similar_movies", []):
            components.similar_card(item)

with profile_tab:
    st.subheader("Taste Profile")
    if st.button("Load Profile"):
        profile = api_client.get(f"/user-profile/{user_id}")
        genres = profile.get("genres", {})
        decades = profile.get("decades", {})
        directors = profile.get("directors", {})
        actors = profile.get("actors", {})

        if genres:
            st.plotly_chart(charts.genre_radar(genres), use_container_width=True, key="genre_radar")
        else:
            st.info("No genre data found for this user.")

        if decades:
            st.plotly_chart(charts.decade_bar(decades), use_container_width=True, key="decade_bar")
        else:
            st.info("No decade data found for this user.")

        if directors:
            st.plotly_chart(charts.top_bar(directors, "Top Directors"), use_container_width=True, key="top_directors")
        else:
            st.info("No director data found (TMDB credits not available).")

        if actors:
            st.plotly_chart(charts.top_bar(actors, "Top Actors"), use_container_width=True, key="top_actors")
        else:
            st.info("No actor data found (TMDB credits not available).")

with insights_tab:
    st.subheader("Model Insights")
    if st.button("Explain Recommendations"):
        data = api_client.post("/explain", {"user_id": user_id, "top_k": top_k})
        for rec in data.get("explanations", []):
            components.recommendation_card(rec)
