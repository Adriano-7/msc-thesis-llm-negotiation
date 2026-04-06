import sys
import os
import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("../")
sys.path.append(".")
from dotenv import load_dotenv

load_dotenv("../runner/.env")

# Try importing LOGS_ROOT, fallback if it fails
try:
    from analysis.common import LOGS_ROOT
except ImportError:
    LOGS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".logs"))

st.set_page_config(
    page_title="MultiAgent Negotiation Explorer",
    page_icon="🤖",
    layout="wide",
)

@st.cache_data(ttl=3600)
def load_log_overview():
    data = []
    logs_path = Path(LOGS_ROOT)
    
    if not logs_path.exists():
        return pd.DataFrame()

    # Fast scan of directories
    for root, dirs, files in os.walk(logs_path):
        if "game_state.json" in files:
            rel_path = Path(root).relative_to(logs_path)
            parts = rel_path.parts
            
            section = "Unknown"
            game = "Unknown"
            retry = "Unknown"
            size = "Unknown"
            pairing = "Unknown"

            if len(parts) >= 2:
                section_raw = parts[0]
                game_raw = parts[1]
                
                if section_raw == "section_one" and len(parts) >= 5:
                    retry = parts[2]
                    size = parts[3]
                    pairing = parts[4]
                elif section_raw == "section_two" and len(parts) >= 4:
                    size = parts[2]
                    pairing = parts[3]
                    retry = "N/A"
                
                # Clean strings for display
                section = section_raw.replace("_", " ").title()
                game = game_raw.replace("_section_one", "").replace("_section_two", "").replace("_personas", "").replace("_", " ").title()
                size = size.replace("_", " ").title()
                
            data.append({
                "Section": section,
                "Game": game,
                "Retry Setting": retry,
                "Model Size": size,
                "Pairing": pairing,
                "Path": str(rel_path)
            })
                
    return pd.DataFrame(data)

with st.spinner("Scanning experiment logs..."):
    df = load_log_overview()

if df.empty:
    st.info(f"No logs found in {LOGS_ROOT}. Please run experiments to generate logs.")
else:
    st.header("Experiment Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Games Played", f"{len(df):,}")
    col2.metric("Sections", df["Section"].nunique())
    col3.metric("Game Types", df["Game"].nunique())
    col4.metric("Model Sizes Evaluated", df["Model Size"].nunique())
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Games Played per Type")
        df_game_counts = df["Game"].value_counts().reset_index()
        df_game_counts.columns = ["Game", "Count"]
        df_game_counts.set_index("Game", inplace=True)
        st.bar_chart(df_game_counts)
        
    with col_right:
        st.subheader("Distribution by Model Size")
        df_size_counts = df["Model Size"].value_counts().reset_index()
        df_size_counts.columns = ["Model Size", "Count"]
        df_size_counts.set_index("Model Size", inplace=True)
        st.bar_chart(df_size_counts)


    
    with st.expander("View Raw Data Summary"):
        st.dataframe(df.drop(columns=["Path"]).head(500), use_container_width=True)
