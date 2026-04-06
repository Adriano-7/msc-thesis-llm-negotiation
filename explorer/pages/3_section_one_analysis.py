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
from analysis.section_one import (
    load_completion_stats,
    load_trading_games,
    load_ultimatum_games,
    load_buysell_games,
    load_retry_data,
    plot_completion_rate_by_pairing,
    plot_completion_rate_by_model,
    plot_trading_heatmap,
    plot_ultimatum_heatmap,
    plot_buysell_bar,
    plot_buysell_heatmap,
    plot_retry_comparison,
    plot_retry_by_pairing,
    plot_retry_distribution,
)

st.set_page_config(page_title="Section 1 Analysis", layout="wide")
st.title("Section 1 Analysis — Baseline Results")

available_sizes = get_available_sizes(section=1)
if not available_sizes:
    st.error(f"No data found in {LOGS_ROOT}/section_one/. Check that experiment logs exist.")
    st.stop()

SIZE_ALL = "__all__"

model_size = st.sidebar.selectbox(
    "Model Size",
    [SIZE_ALL] + available_sizes,
    format_func=lambda x: "All" if x == SIZE_ALL else x.replace("_", " ").title(),
)

sizes_to_show = available_sizes if model_size == SIZE_ALL else [model_size]

tab_completion, tab_trading, tab_ultimatum, tab_buysell, tab_retry = st.tabs(
    ["Completion Rates", "Trading Game", "Ultimatum Game", "Buy-Sell Game", "Retry Ablation"]
)

# Completion Rates 
with tab_completion:
    st.subheader("By Model Pairing")
    cols = st.columns(len(sizes_to_show))
    for col, size in zip(cols, sizes_to_show):
        with col:
            if len(sizes_to_show) > 1:
                st.markdown(f"**{size.replace('_', ' ').title()}**")
            with st.spinner(f"Loading {size}..."):
                df_comp, agents_clean = load_completion_stats(LOGS_ROOT, size)
            if df_comp.empty:
                st.warning("No data.")
            else:
                fig = plot_completion_rate_by_pairing(df_comp, size, agents_clean)
                st.pyplot(fig)
                plt.close(fig)

    st.subheader("By Individual Model")
    cols = st.columns(len(sizes_to_show))
    for col, size in zip(cols, sizes_to_show):
        with col:
            if len(sizes_to_show) > 1:
                st.markdown(f"**{size.replace('_', ' ').title()}**")
            df_comp, agents_clean = load_completion_stats(LOGS_ROOT, size)
            if not df_comp.empty:
                fig = plot_completion_rate_by_model(df_comp, size, agents_clean)
                st.pyplot(fig)
                plt.close(fig)

# Trading Game
with tab_trading:
    st.subheader("Win Rate & Payoff Heatmaps")
    cols = st.columns(len(sizes_to_show))
    for col, size in zip(cols, sizes_to_show):
        with col:
            if len(sizes_to_show) > 1:
                st.markdown(f"**{size.replace('_', ' ').title()}**")
            with st.spinner(f"Loading {size}..."):
                df_trading, agents_clean = load_trading_games(LOGS_ROOT, size)
            if df_trading.empty:
                st.warning("No data.")
            else:
                fig = plot_trading_heatmap(df_trading, size, agents_clean)
                st.pyplot(fig)
                plt.close(fig)

# Ultimatum Game
with tab_ultimatum:
    st.subheader("Win Rate & Proposer Payoff Heatmaps")
    cols = st.columns(len(sizes_to_show))
    for col, size in zip(cols, sizes_to_show):
        with col:
            if len(sizes_to_show) > 1:
                st.markdown(f"**{size.replace('_', ' ').title()}**")
            with st.spinner(f"Loading {size}..."):
                df_ult, agents_clean = load_ultimatum_games(LOGS_ROOT, size)
            if df_ult.empty:
                st.warning("No data.")
            else:
                fig = plot_ultimatum_heatmap(df_ult, size, agents_clean)
                st.pyplot(fig)
                plt.close(fig)

# Buy-Sell Game─
with tab_buysell:
    st.subheader("Buyer Outcome — Seller at 40, Buyer at 60")
    cols = st.columns(len(sizes_to_show))
    for col, size in zip(cols, sizes_to_show):
        with col:
            if len(sizes_to_show) > 1:
                st.markdown(f"**{size.replace('_', ' ').title()}**")
            with st.spinner(f"Loading {size}..."):
                df_bs, agents_clean = load_buysell_games(LOGS_ROOT, size)
            if df_bs.empty:
                st.warning("No data.")
            else:
                fig = plot_buysell_bar(df_bs, size, agents_clean)
                st.pyplot(fig)
                plt.close(fig)

    st.subheader("Seller & Buyer Profit Heatmaps")
    cols = st.columns(len(sizes_to_show))
    for col, size in zip(cols, sizes_to_show):
        with col:
            if len(sizes_to_show) > 1:
                st.markdown(f"**{size.replace('_', ' ').title()}**")
            df_bs, agents_clean = load_buysell_games(LOGS_ROOT, size)
            if not df_bs.empty:
                fig = plot_buysell_heatmap(df_bs, size, agents_clean)
                st.pyplot(fig)
                plt.close(fig)

# Retry Ablation
with tab_retry:
    st.subheader("Baseline vs Self-Correction Completion")
    cols = st.columns(len(sizes_to_show))
    for col, size in zip(cols, sizes_to_show):
        with col:
            if len(sizes_to_show) > 1:
                st.markdown(f"**{size.replace('_', ' ').title()}**")
            with st.spinner(f"Loading {size}..."):
                df_retry, agents_clean = load_retry_data(LOGS_ROOT, size)
            if df_retry.empty:
                st.info("No retry data.")
            else:
                fig = plot_retry_comparison(df_retry, size)
                st.pyplot(fig)
                plt.close(fig)

    st.subheader("Completion by Pairing")
    for size in sizes_to_show:
        if len(sizes_to_show) > 1:
            st.markdown(f"**{size.replace('_', ' ').title()}**")
        df_retry, _ = load_retry_data(LOGS_ROOT, size)
        if not df_retry.empty:
            for game_name, fig in plot_retry_by_pairing(df_retry, size):
                st.write(f"*{game_name}*")
                st.pyplot(fig)
                plt.close(fig)

    st.subheader("Retry Distribution")
    cols = st.columns(len(sizes_to_show))
    for col, size in zip(cols, sizes_to_show):
        with col:
            if len(sizes_to_show) > 1:
                st.markdown(f"**{size.replace('_', ' ').title()}**")
            df_retry, _ = load_retry_data(LOGS_ROOT, size)
            if not df_retry.empty:
                fig = plot_retry_distribution(df_retry, size)
                st.pyplot(fig)
                plt.close(fig)
