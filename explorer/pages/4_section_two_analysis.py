import sys
import os

sys.path.append("../")
sys.path.append(".")

import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore")

sns.set_context("poster")
sns.set_palette("colorblind")
matplotlib.rcParams.update({"font.size": 22})
sns.set(font_scale=2.4)

from analysis.common import LOGS_ROOT, get_available_sizes
from analysis.section_two import (
    load_trading_games_s2,
    load_ultimatum_games_s2,
    load_buysell_games_s2,
    load_completion_stats_s2,
    plot_trading_persona_per_model,
    plot_trading_persona_combined,
    plot_ultimatum_persona_per_model,
    plot_ultimatum_persona_combined,
    plot_ultimatum_no_deal_rate,
    plot_buysell_persona_per_model,
    plot_buysell_persona_combined,
    plot_completion_by_persona,
)

st.set_page_config(page_title="Section 2 Persona Analysis", layout="wide")
st.title("Section 2 Analysis — Persona Effects")

available_sizes = get_available_sizes(section=2)
if not available_sizes:
    st.error(f"No Section 2 data found in {LOGS_ROOT}/section_two/.")
    st.stop()

model_size = st.sidebar.selectbox(
    "Model Size",
    available_sizes,
    format_func=lambda x: x.replace("_", " ").title(),
)

tab_trading, tab_ultimatum, tab_buysell, tab_completion = st.tabs(
    ["Trading Game", "Ultimatum Game", "Buy-Sell Game", "Completion by Persona"]
)

#  Trading Persona 
with tab_trading:
    with st.spinner("Loading trading data..."):
        df_trading, agents_clean = load_trading_games_s2(LOGS_ROOT, model_size)

    if df_trading.empty:
        st.warning("No trading data found for this model size.")
    else:
        st.subheader("Per-Model Breakdown")
        for model_name in agents_clean:
            sub = df_trading[df_trading["model_1"] == model_name]
            if sub.empty:
                continue
            with st.expander(f"{model_name}", expanded=False):
                fig = plot_trading_persona_per_model(df_trading, model_name, model_size)
                st.pyplot(fig)
                plt.close(fig)

        st.subheader("Combined — All Models")
        fig = plot_trading_persona_combined(df_trading, model_size)
        st.pyplot(fig)
        plt.close(fig)

#  Ultimatum Persona 
with tab_ultimatum:
    with st.spinner("Loading ultimatum data..."):
        df_ult, agents_clean = load_ultimatum_games_s2(LOGS_ROOT, model_size)

    if df_ult.empty:
        st.warning("No ultimatum data found for this model size.")
    else:
        st.subheader("Per-Model Breakdown")
        for model_name in agents_clean:
            sub = df_ult[df_ult["model_1"] == model_name]
            if sub.empty:
                continue
            with st.expander(f"{model_name}", expanded=False):
                fig = plot_ultimatum_persona_per_model(df_ult, model_name, model_size)
                st.pyplot(fig)
                plt.close(fig)

        st.subheader("Combined — All Models")
        fig = plot_ultimatum_persona_combined(df_ult, model_size)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("No-Deal Rate by Persona")
        fig = plot_ultimatum_no_deal_rate(df_ult, model_size)
        st.pyplot(fig)
        plt.close(fig)

#  BuySell Persona 
with tab_buysell:
    with st.spinner("Loading buy-sell data..."):
        df_bs, agents_clean = load_buysell_games_s2(LOGS_ROOT, model_size)

    if df_bs.empty:
        st.warning("No buy-sell data found for this model size.")
    else:
        st.subheader("Per-Model Breakdown")
        for model_name in agents_clean:
            sub = df_bs[df_bs["model_1"] == model_name]
            if sub.empty:
                continue
            with st.expander(f"{model_name}", expanded=False):
                fig = plot_buysell_persona_per_model(df_bs, model_name, model_size)
                st.pyplot(fig)
                plt.close(fig)

        st.subheader("Combined — All Models")
        fig = plot_buysell_persona_combined(df_bs, model_size)
        st.pyplot(fig)
        plt.close(fig)

#  Completion by Persona 
with tab_completion:
    with st.spinner("Loading completion data..."):
        df_comp = load_completion_stats_s2(LOGS_ROOT, model_size)

    if df_comp.empty:
        st.warning("No completion data found for this model size.")
    else:
        fig = plot_completion_by_persona(df_comp, model_size)
        st.pyplot(fig)
        plt.close(fig)
