import sys

sys.path.append("../")
sys.path.append(".")

import openai
from dotenv import load_dotenv
load_dotenv("../../runner/.env")

import os
os.environ["OPENAI_API_KEY"] = "g"

import os
import json
from glob import glob
from utils import *
import streamlit as st
from ratbench.constants import *
from explorer.basic_elements.game_filtering import *
from games import *

st.set_page_config(page_title="Conversation Explorer", layout="wide")

# main page
st.write("# Conversation Explorer")

# data loading
root_dir = os.path.abspath(__file__).split("/")[:-3]
base_logs = os.path.join("/", *root_dir, ".logs")

with st.expander("Control Panel: Data Source & Filtering", expanded=True):
    st.markdown("##### 1. Select Log Directory")
    if os.path.isdir(base_logs):
        col1, col2, col3, col4 = st.columns(4)

        sections = sorted([d for d in os.listdir(base_logs) if os.path.isdir(os.path.join(base_logs, d))])
        with col1:
            selected_section = st.selectbox("Section", sections)

        section_path = os.path.join(base_logs, selected_section)
        game_types = sorted([d for d in os.listdir(section_path) if os.path.isdir(os.path.join(section_path, d))])
        with col2:
            selected_game_type = st.selectbox(
                "Game Type",
                game_types,
                format_func=lambda x: x.split("_")[0].capitalize(),
            )

        game_path = os.path.join(section_path, selected_game_type)
        variants = sorted([d for d in os.listdir(game_path) if os.path.isdir(os.path.join(game_path, d))])
        with col3:
            selected_variant = st.selectbox("Variant", variants)

        variant_path = os.path.join(game_path, selected_variant)
        model_sizes = sorted([d for d in os.listdir(variant_path) if os.path.isdir(os.path.join(variant_path, d))])
        with col4:
            selected_model_size = st.selectbox("Model Size", ["All"] + model_sizes)

        log_dir = variant_path if selected_model_size == "All" else os.path.join(variant_path, selected_model_size)
    else:
        log_dir = st.text_input("Log Directory", value=base_logs)

    st.caption(f"Loading from: `{log_dir}`")

    with st.spinner("Loading games..."):
        games = load_states_from_dir(log_dir, completed_only=False)

    if games:
        games_summary_df = compute_game_summary(games)
        games_summary_df["list_name"] = games_summary_df[["game_name", "log_path", "is_complete"]].apply(
            lambda row: f"{'✓' if row.is_complete else '✗'} {row.game_name} - {from_timestamp_str(os.path.basename(row.log_path))} - {str(os.path.basename(row.log_path))}",
            axis=1,
        )

        st.markdown("---")
        games_summary_df = game_filter(games_summary_df)

if games:
    if games_summary_df.empty:
        st.warning("No games match the current filters.")
    else:
        st.markdown("### Select a Conversation")
        col_game, col_view = st.columns([3, 1])
        with col_game:
            selected_game = st.selectbox("Which Game?", list(games_summary_df["list_name"]))
        with col_view:
            view_option = st.selectbox("View Mode", ["Unified Timeline", "Player 1 Perspective", "Player 2 Perspective"])

        game_to_load = get_log_path_from_summary(selected_game, games_summary_df)

        with open(game_to_load) as f:
            # Load the json file
            game_state = json.load(f)

        st.markdown("---")
        
        if view_option == "Unified Timeline":
            st.write("**Unified Conversation View**")
            
            p1 = game_state["players"][0]
            p2 = game_state["players"][1]
            
            p1_name = p1.get("agent_name", "Player 1")
            p2_name = p2.get("agent_name", "Player 2")
            
            # Show System Prompts side-by-side
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                with st.expander(f"Check System Prompt - {p1_name}"):
                    sys_prompt_txt = p1["conversation"][0]["content"]
                    st.write(text_formatting(sys_prompt_txt, True))
            
            with col_p2:
                with st.expander(f"Check System Prompt - {p2_name}"):
                    sys_prompt_txt = p2["conversation"][0]["content"]
                    st.write(text_formatting(sys_prompt_txt, True))
            
            st.markdown("---")
            
            # Gather assistant messages
            p1_conv = [msg for msg in p1["conversation"] if msg["role"] == "assistant"]
            p2_conv = [msg for msg in p2["conversation"] if msg["role"] == "assistant"]
            
            max_len = max(len(p1_conv), len(p2_conv))
            
            red_avatar = os.path.join(os.path.dirname(os.path.dirname(__file__)), "red_robot.svg")
            blue_avatar = os.path.join(os.path.dirname(os.path.dirname(__file__)), "blue_robot.svg")
            
            for i in range(max_len):
                if i < len(p1_conv):
                    with st.chat_message(p1_name, avatar=red_avatar):
                        st.write(text_formatting(p1_conv[i]["content"], False))
                
                if i < len(p2_conv):
                    with st.chat_message(p2_name, avatar=blue_avatar):
                        st.write(text_formatting(p2_conv[i]["content"], False))
                        
        else:
            option = int(view_option.split(" ")[1])
            st.write(f"**You are looking at Player {option}'s view**")
            
            red_avatar = os.path.join(os.path.dirname(os.path.dirname(__file__)), "red_robot.svg")
            blue_avatar = os.path.join(os.path.dirname(os.path.dirname(__file__)), "blue_robot.svg")
            
            for index, msg in enumerate(game_state["players"][option - 1]["conversation"]):
                txtmsg = msg["content"]
                sys_prompt = True if index == 0 else False

                txtmsg = text_formatting(txtmsg, sys_prompt)

                if sys_prompt:
                    with st.expander("Check System Prompt"):
                        with st.chat_message(msg["role"]):
                            st.write(txtmsg)
                else:
                    avatar = None
                    if msg["role"] == "assistant":
                        avatar = red_avatar if option == 1 else blue_avatar
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.write(txtmsg)
else:
    st.info("No games found in the selected directory.")
