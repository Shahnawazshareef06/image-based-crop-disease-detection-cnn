import streamlit as st

st.set_page_config(page_title="TerraVista", page_icon="🌿", layout="wide")
st.markdown("""
<style>
[data-testid="stMainBlockContainer"] {max-width: 74rem; padding-bottom: 4rem;}
.tv-brand {color: var(--st-primary-color); font-size: 1.05rem; font-weight: 700;
           letter-spacing: .015em; margin: .35rem 0 -.5rem;}
h1 {font-family: Georgia, serif !important; font-weight: 500 !important;
    letter-spacing: -.035em;}
.tv-specimen {position: relative; margin: 0; overflow: hidden; border-radius: 14px;
              background: #233c31; border: 1px solid #385147;}
.tv-specimen img {display: block; width: 100%; height: clamp(12rem, 22vw, 17rem);
                  object-fit: contain; background: #242f28;}
.tv-specimen figcaption {padding: .8rem 1rem; color: #c8dacc; font-size: .85rem;
                         display: flex; align-items: center; justify-content: space-between;}
.tv-specimen strong {color: #f0f6eb; font-size: 1rem;}
</style>
""", unsafe_allow_html=True)
page = st.navigation([
    st.Page("pages/leaf.py", title="Leaf analysis", icon=":material/eco:", default=True),
    st.Page("pages/1_Results.py", title="Model results", icon=":material/bar_chart:", url_path="Results"),
], position="top")
page.run()
