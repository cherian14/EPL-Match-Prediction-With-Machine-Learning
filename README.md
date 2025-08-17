# EPL Predictor Arena 🏟️

### *Harnessing the power of data to predict the beautiful game. An end-to-end Python project that forecasts English Premier League match winners using machine learning and an interactive Streamlit UI.*

<br>

![Streamlit App Demo](https://i.imgur.com/your-demo-gif-url.gif) 
<!-- **Pro Tip:** Record a short GIF of you using the Streamlit app and upload it. Replace the URL above. A live demo is the best way to impress. -->

<p align="center">
  <a href="your-live-streamlit-app-url" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App" height="30">
  </a>
</p>

---

## The Challenge: Decoding the Unpredictable

The English Premier League is a theater of dreams, a weekly spectacle of unpredictable drama. For fans, it's passion; for data scientists, it's a high-stakes challenge. This project steps into that arena, attempting to cut through the noise and transform raw match data into **predictive insight**. The goal isn't just to build a model, but to engineer an intelligent system that understands team form and tactical nuances to forecast a victor.

## Core Features

-   ** Automated Data Scraping**: Deploys `requests` and `BeautifulSoup` to scout FBref.com, gathering multi-season match and performance statistics.
-   ** Advanced Feature Engineering**: Goes beyond simple stats by creating **rolling averages** to dynamically model a team's current form.
-   ** Intelligent Prediction Engine**: Utilizes a `RandomForestClassifier` to learn complex patterns and predict match outcomes with a focus on high-precision results.
-   ** Cinematic User Interface**: A sleek, user-friendly front-end built with `Streamlit`, allowing anyone to become a data-driven pundit.
-   ** High-Confidence Analysis**: Implements a unique logic to identify high-probability predictions by cross-referencing predictions for both competing teams.

---

## The Technology Stack

This project was built with a modern, powerful Python data science stack.

| Technology      | Purpose                                    |
| :-------------- | :----------------------------------------- |
| **Python**      | Core programming language                  |
| **Pandas**      | Data manipulation, cleaning, and analysis  |
| **Scikit-learn**| Machine learning model implementation      |
| **Requests**    | Web scraping (HTTP requests)               |
| **BeautifulSoup**| HTML parsing for data extraction           |
| **Streamlit**   | Creating the interactive web application   |

---

## The Playbook: Project Methodology

The project follows a five-phase strategic plan, from data acquisition to final prediction.

### Phase 1: The Scout - Data Acquisition
The initial phase involved deploying a web scraper to gather comprehensive match data. The script navigates through multiple seasons of EPL stats on FBref, extracting everything from basic scores to detailed shooting statistics for every team.

> **Outcome:** A robust `matches.csv` dataset, forming the bedrock of our analysis.

### Phase 2: Pre-Match Prep - Data Cleansing & Feature Engineering
Raw data is refined. Dates are standardized, and categorical data like team names and venues are converted into numerical codes. New, insightful features are engineered, such as `hour` of the match and `day_of_week`, to capture hidden temporal patterns.

> **Outcome:** A clean, model-ready dataset with foundational predictive features.

### Phase 3: The Tactician - Advanced Feature Engineering
This is where the model gains its strategic edge. Instead of treating each match in isolation, we calculate **rolling averages** for crucial performance metrics (goals for/against, shots on target, etc.) over the previous three games. This gives the model a crucial understanding of a team's current **form and momentum**.

> **Outcome:** A feature-rich dataset that provides historical context for each match.

### Phase 4: The Oracle - Predictive Modeling
A **Random Forest Classifier** was chosen for its ability to handle complex, non-linear relationships in the data. The model was trained on historical data and rigorously evaluated using **Precision Score** to ensure that when it predicts a win, it is correct as often as possible.

> **Outcome:** A trained model capable of forecasting match outcomes with an initial precision of **62%**, rising to over **67%** for high-confidence predictions.

### Phase 5: The Grandstand - Interactive UI
The final model is deployed in a cinematic, user-friendly web application using **Streamlit**. This interface allows anyone to input a hypothetical match-up, select a date and time, and receive an instant, data-driven prediction, complete with win probabilities and a clear verdict.

> **Outcome:** A shareable, interactive tool that brings the power of the prediction engine to everyone.

---

## Get in the Game: Installation & Usage

Step into the arena yourself. Follow these steps to run the predictor on your local machine.

**1. Clone the Repository:**
```bash
git clone https://github.com/your-username/EPL-Predictor-Arena.git
cd EPL-Predictor-Arena
```

**2. Install Dependencies:**
```bash
pip install -r requirements.txt
# (Ensure you have a requirements.txt file with streamlit, pandas, scikit-learn)
```

**3. Run the Application:**
The script will automatically use the included `matches.csv`.
```bash
streamlit run app.py
```

Your browser will open with the EPL Predictor Arena, ready for your first prediction!

---

## The Next Season: Future Enhancements

The arena is never static. Future developments could include:

-   **Deeper Player Stats:** Integrating player-specific data (injuries, form, ELO ratings).
-   **Alternative Models:** Experimenting with Gradient Boosting models (like XGBoost) or Neural Networks for potentially higher accuracy.
-   **Real-Time Odds Integration:** Comparing model predictions against live betting odds to identify value.
-   **Expanded Data:** Scraping data from over a decade of EPL matches to build an even more robust model.

---

## 🤝 Connect with Me

**Built with ❤️ by Cherian R**

*   **GitHub:** [@cherian14](https://github.com/cherian14)
*   **LinkedIn:** [Cherian R](https://www.linkedin.com/in/cherian-r-a1bba3292/)
*   **Email:** `cherian262005@gmail.com`

Feel free to reach out with any questions, suggestions, or collaboration ideas!
