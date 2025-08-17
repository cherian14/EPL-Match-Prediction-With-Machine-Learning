# app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="EPL Predictor Arena",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Aesthetic Themes & Custom CSS ---
def load_css():
    """Injects custom CSS to style the Streamlit app for a cinematic look."""
    
    # --- CHOOSE YOUR THEME ---
    # Uncomment one of the themes below to change the look and feel.
    
    # Theme 1: Spotify - Modern & Energetic
    primary_color = "#1DB954" # Vibrant Green
    background_color = "#191414" # Deep Black/Grey
    secondary_bg_color = "#221F1F"
    text_color = "#FFFFFF"
    secondary_text_color = "#B3B3B3"

    # # Theme 2: Netflix - Dramatic & Powerful
    # primary_color = "#E50914" # Netflix Red
    # background_color = "#141414" # Near Black
    # secondary_bg_color = "#221F1F"
    # text_color = "#FFFFFF"
    # secondary_text_color = "#B3B3B3"
    
    # # Theme 3: Airbnb - Welcoming & Friendly
    # primary_color = "#FF385C" # Soft Coral
    # background_color = "#222222" # Dark Grey
    # secondary_bg_color = "#484848"
    # text_color = "#FFFFFF"
    # secondary_text_color = "#EBEBEB"

    css = f"""
    <style>
        /* --- General App Styling --- */
        .stApp {{
            background-color: {background_color};
            color: {text_color};
        }}
        
        /* --- Headers --- */
        h1, h2, h3 {{
            color: {primary_color};
            font-weight: 700 !important;
        }}

        /* --- Input Widgets --- */
        .stSelectbox div[data-baseweb="select"] > div, .stDateInput div[data-baseweb="base-input"], .stTimeInput div[data-baseweb="base-input"] {{
            background-color: {secondary_bg_color};
            border: 1px solid {secondary_bg_color};
            border-radius: 10px;
        }}

        /* --- Main Button --- */
        .stButton > button {{
            background-color: {primary_color};
            color: {background_color};
            border: none;
            font-weight: bold;
            border-radius: 25px;
            padding: 12px 30px;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .stButton > button:hover {{
            background-color: {text_color};
            color: {background_color};
            transform: scale(1.05);
        }}
        .stButton > button:focus {{
            box-shadow: 0 0 0 2px {background_color}, 0 0 0 4px {primary_color};
        }}

        /* --- Metric Cards for Results --- */
        .stMetric {{
            background-color: {secondary_bg_color};
            border-radius: 10px;
            padding: 20px;
            border-left: 7px solid {primary_color};
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .stMetric .stMetricLabel {{
            color: {secondary_text_color};
            font-size: 1rem;
        }}
        .stMetric .stMetricValue {{
            font-size: 2.5rem;
            color: {text_color};
        }}

        /* --- Result Alerts --- */
        .stAlert {{
            border-radius: 10px;
            border: none;
        }}

        /* --- Expander for 'How it Works' --- */
        .stExpander {{
            background-color: {secondary_bg_color};
            border: 1px solid {primary_color};
            border-radius: 10px;
        }}
        
        /* --- Progress Bar --- */
        .stProgressBar > div > div > div > div {{
            background-color: {primary_color};
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Caching Functions ---
@st.cache_data
def load_and_prepare_data(data_path="matches.csv"):
    # (This function remains the same as before)
    if not os.path.exists(data_path):
        st.error(f"Error: '{data_path}' not found. Please ensure the dataset is in the same folder.")
        st.stop()
    matches = pd.read_csv(data_path, index_col=0)
    if "comp" in matches.columns: del matches["comp"]
    if "notes" in matches.columns: del matches["notes"]
    matches["date"] = pd.to_datetime(matches["date"])
    matches["target"] = (matches["result"] == "W").astype("int")
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
    matches["day_code"] = matches["date"].dt.dayofweek
    def rolling_averages(group, cols, new_cols):
        group = group.sort_values("date")
        rolling_stats = group[cols].rolling(3, closed='left').mean()
        group[new_cols] = rolling_stats
        group = group.dropna(subset=new_cols)
        return group
    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
    new_cols = [f"{c}_rolling" for c in cols]
    matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
    matches_rolling = matches_rolling.droplevel('team')
    matches_rolling.index = range(matches_rolling.shape[0])
    team_opp_code_map = dict(zip(matches['team'], matches['opp_code']))
    return matches_rolling, team_opp_code_map, new_cols

@st.cache_resource
def train_model(data, predictors):
    # (This function remains the same as before)
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    rf.fit(data[predictors], data["target"])
    return rf

# --- Load Data, CSS, and Train Model ---
load_css()
try:
    full_data, team_map, new_cols = load_and_prepare_data()
    predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols
    model = train_model(full_data, predictors)
except Exception as e:
    st.error(f"An error occurred during data loading or model training: {e}")
    st.stop()

# --- UI Layout ---
st.markdown("""
<div style="text-align: center;">
    <h1 style="font-size: 3.5rem; font-weight: 700; letter-spacing: -2px;">EPL PREDICTOR ARENA</h1>
    <p style="font-size: 1.2rem; color: #B3B3B3; margin-top: -10px;">WHERE DATA DECIDES THE VICTOR</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Match Setup Container ---
with st.container():
    st.header("MATCH SETUP")
    teams = sorted(full_data["team"].unique())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Select Home Team", options=teams, index=teams.index("Manchester City"))
    with col2:
        away_team_options = [team for team in teams if team != home_team]
        away_team = st.selectbox("Select Away Team", options=away_team_options, index=away_team_options.index("Liverpool"))
        
    col3, col4 = st.columns(2)
    with col3:
        match_date = st.date_input("Match Date", datetime.now())
    with col4:
        match_time = st.time_input("Match Time (HH:MM)", datetime.strptime("15:00", "%H:%M").time())

# --- Prediction Button ---
if st.button("🔥 GENERATE PREDICTION", use_container_width=True):
    with st.spinner("Analyzing historical data..."):
        try:
            home_stats = full_data[full_data["team"] == home_team].sort_values("date").iloc[-1]
            away_stats = full_data[full_data["team"] == away_team].sort_values("date").iloc[-1]
        except IndexError:
            st.error("Model cannot make a prediction. Not enough recent match data for one of the teams.")
            st.stop()

        home_pred_data = {"venue_code": 1, "opp_code": team_map.get(away_team), "hour": match_time.hour, "day_code": match_date.weekday()}
        away_pred_data = {"venue_code": 0, "opp_code": team_map.get(home_team), "hour": match_time.hour, "day_code": match_date.weekday()}
        for col in new_cols:
            home_pred_data[col] = home_stats[col]
            away_pred_data[col] = away_stats[col]
        home_df = pd.DataFrame([home_pred_data], columns=predictors)
        away_df = pd.DataFrame([away_pred_data], columns=predictors)
        
        home_pred_prob = model.predict_proba(home_df)[0][1]
        away_pred_prob = model.predict_proba(away_df)[0][1]
        home_pred, away_pred = model.predict(home_df)[0], model.predict(away_df)[0]
        
        # --- Display Results ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.header("PREDICTION RESULTS")
        
        # Win Probability Metrics
        col_res1, col_res2 = st.columns(2)
        col_res1.metric(label=f"{home_team} Win Probability", value=f"{home_pred_prob:.1%}")
        col_res2.metric(label=f"{away_team} Win Probability", value=f"{away_pred_prob:.1%}")
        
        # Probability Bar
        st.write("Probability Balance")
        progress_value = int(home_pred_prob * 100)
        st.progress(progress_value)
        st.caption(f"{home_team} ({home_pred_prob:.1%}) vs {away_team} ({away_pred_prob:.1%})")

        st.markdown("---")
        st.subheader("MODEL'S VERDICT")
        
        # Final Verdict Logic
        if home_pred == 1 and away_pred == 0:
            st.success(f"🏆 **WINNER: {home_team}**")
            st.write(f"The model has high confidence in a **{home_team}** victory. Their recent performance and home advantage create a strong outlook.")
        elif home_pred == 0 and away_pred == 1:
            st.success(f"🏆 **WINNER: {away_team}**")
            st.write(f"The model confidently predicts an away win for **{away_team}**. Their current form appears to overcome the home-field disadvantage.")
        else:
            st.warning("⚖️ **UNCERTAIN OUTCOME / LIKELY DRAW**")
            if home_pred == 0 and away_pred == 0:
                 st.write("Neither team is predicted to win, making a **DRAW** the most probable result. Expect a tight, defensive match.")
            else:
                 st.write("The model's signals are conflicting, suggesting a highly unpredictable match. While a draw is very possible, the team with the higher probability has a slight edge.")

# --- Explainer Section ---
st.markdown("<hr style='border: 1px solid #484848;'>", unsafe_allow_html=True)
with st.expander(" curious? See How This Predictor Works"):
    st.markdown("""
        This tool uses a machine learning model called a **Random Forest** to predict match outcomes.
        
        - **Training Data:** It learned patterns from thousands of EPL matches from the 2020-2022 seasons.
        - **Key Factors:** The model considers more than just wins and losses. It analyzes:
            - **Venue:** Home or Away advantage.
            - **Recent Form:** It calculates *rolling averages* of key stats (like goals, shots, and goals against) from the last 3 games.
            - **Opponent Strength:** The historical performance of the opponent.
            - **Match Time:** The day of the week and hour of the match.
        
        *Disclaimer: This is an educational tool. It does not account for real-time factors like player injuries or team news and is not a guarantee of future results.*
    """)

st.markdown("<hr style='border: 1px solid #484848;'>", unsafe_allow_html=True)
st.caption("EPL Predictor Arena | A Data-Driven Approach to Football Forecasting")