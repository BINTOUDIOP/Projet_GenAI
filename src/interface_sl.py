import streamlit as st
from router import answer

st.set_page_config(page_title="Assistant Multi-Compétences", page_icon="🤖")

st.title("🤖 Assistant Intelligent (RAG + Agents)")
st.caption("• RAG sur tes documents • Outils (calcul, météo, web) • Mémoire de conversation")

if "history" not in st.session_state:
    st.session_state.history = []

def add_message(role, content):
    st.session_state.history.append({"role": role, "content": content})

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Pose ta question…")
if user_input:
    add_message("user", user_input)
    with st.chat_message("assistant"):
        with st.spinner("Réflexion…"):
            resp = answer(user_input)
            st.markdown(resp)
    add_message("assistant", resp)

st.sidebar.header("⚙️ Utilisation")
st.sidebar.markdown("""
1. Place tes **PDF/DOCX** dans `./data/`.
2. Lance `python src/ingest.py` pour **indexer**.
3. Démarre l'app: `streamlit run src/interface_sl.py`.
""")