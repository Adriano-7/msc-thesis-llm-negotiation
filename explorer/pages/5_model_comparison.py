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
    plot_completion_rate_by_pairing,
    plot_completion_rate_by_model,
    plot_trading_heatmap,
    plot_ultimatum_heatmap,
    plot_buysell_bar,
    plot_buysell_heatmap,
)

st.set_page_config(page_title="Model Size Comparison", layout="wide")
st.title("Model Size Comparison")
st.caption("Select two model sizes to compare their results side by side.")

available_sizes = get_available_sizes(section=1)
if len(available_sizes) < 2:
    st.warning("Need at least 2 model sizes with data to compare. Currently only have: "
               + str(available_sizes))
    st.stop()

col_a, col_b = st.columns(2)
with col_a:
    size_a = st.selectbox("Model Size A", available_sizes,
                          index=0, format_func=lambda x: x.replace("_", " ").title(),
                          key="size_a")
with col_b:
    size_b = st.selectbox("Model Size B", available_sizes,
                          index=min(1, len(available_sizes) - 1),
                          format_func=lambda x: x.replace("_", " ").title(),
                          key="size_b")

plot_type = st.radio(
    "Which analysis to compare?",
    ["Completion Rate (by Pairing)", "Completion Rate (by Model)", "Trading Game", "Ultimatum Game",
     "Buy-Sell Bar", "Buy-Sell Heatmap"],
    horizontal=True,
)

st.divider()

col1, col2 = st.columns(2)


def render_plot(size: str, col):
    with col:
        st.subheader(size.replace("_", " ").title())
        with st.spinner(f"Loading {size}..."):
            if plot_type == "Completion Rate (by Pairing)":
                df, agents = load_completion_stats(LOGS_ROOT, size)
                fig = plot_completion_rate_by_pairing(df, size, agents) if not df.empty else None
            elif plot_type == "Completion Rate (by Model)":
                df, agents = load_completion_stats(LOGS_ROOT, size)
                fig = plot_completion_rate_by_model(df, size, agents) if not df.empty else None
            elif plot_type == "Trading Game":
                df, agents = load_trading_games(LOGS_ROOT, size)
                fig = plot_trading_heatmap(df, size, agents) if not df.empty else None
            elif plot_type == "Ultimatum Game":
                df, agents = load_ultimatum_games(LOGS_ROOT, size)
                fig = plot_ultimatum_heatmap(df, size, agents) if not df.empty else None
            elif plot_type == "Buy-Sell Bar":
                df, agents = load_buysell_games(LOGS_ROOT, size)
                fig = plot_buysell_bar(df, size, agents) if not df.empty else None
            elif plot_type == "Buy-Sell Heatmap":
                df, agents = load_buysell_games(LOGS_ROOT, size)
                fig = plot_buysell_heatmap(df, size, agents) if not df.empty else None
            else:
                fig = None

        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning(f"No data available for {size}")


render_plot(size_a, col1)
render_plot(size_b, col2)
