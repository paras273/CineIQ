import streamlit as st


def recommendation_card(item):
    st.subheader(item["title"])
    st.write("Score:", round(item["score"], 4))
    if item.get("reasons"):
        st.write("Reasons:")
        for reason in item["reasons"]:
            st.write("-", reason)
    if item.get("score_breakdown"):
        st.write("Score breakdown:", item["score_breakdown"])


def similar_card(item):
    st.write(f"{item['title']} (score: {round(item['score'], 4)})")
