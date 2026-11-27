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
[data-testid="stMainBlockContainer"] {
    padding-inline: clamp(1rem, 4vw, 3.5rem);
    padding-bottom: clamp(2rem, 5vw, 4rem);
}
[data-testid="stMainBlockContainer"] h1 {font-size: clamp(2.15rem, 4vw, 3.4rem);}
[data-testid="stColumn"] {min-width: 0;}
.tv-specimen figcaption {gap: .75rem; flex-wrap: wrap;}
.tv-specimen img {height: clamp(10rem, 18vw, 14rem);}
.tv-table {width: 100%; border-collapse: collapse; font-size: .94rem;}
.tv-table th, .tv-table td {text-align: left; padding: .85rem .65rem;
                          border-bottom: 1px solid var(--st-border-color, #385147);}
.tv-table th {font-size: .86rem;}
.tv-table td {font-variant-numeric: tabular-nums;}
@media (max-width: 52rem) {
    .st-key-intake [data-testid="stHorizontalBlock"] {flex-direction: column; gap: .8rem;}
    .st-key-intake [data-testid="stColumn"] {width: 100%; flex: 1 1 auto;}
    .tv-specimen {display: flex; align-items: center;}
    .tv-specimen img {width: 5.5rem; height: 5.5rem; flex: none;}
    .tv-specimen figcaption {align-items: flex-start; flex-direction: column; gap: .2rem;}
}
@media (max-width: 40rem) {
    [data-testid="stMainBlockContainer"] {padding-inline: 1rem;}
    .tv-table, .tv-table tbody, .tv-table tr {display: block;}
    .tv-table thead {position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%);}
    .tv-table tr {margin: .3rem 0 .9rem; border: 1px solid var(--st-border-color, #385147);
                  border-radius: .8rem; padding: .45rem .8rem;}
    .tv-table td {display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr);
                  gap: .75rem; text-align: right; padding: .48rem 0; overflow-wrap: anywhere;}
    .tv-table td::before {content: attr(data-label); text-align: left; opacity: .8;}
    .tv-table td:last-child {border-bottom: none;}
    [data-testid="stMainBlockContainer"] button {min-height: 2.75rem;}
}
</style>
""", unsafe_allow_html=True)
page = st.navigation([
    st.Page("pages/leaf.py", title="Leaf analysis", icon=":material/eco:", default=True),
    st.Page("pages/1_Results.py", title="Model results", icon=":material/bar_chart:", url_path="Results"),
], position="top")
page.run()
