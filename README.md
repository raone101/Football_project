# Match Outcome Analysis with Machine Learning
This project leverages machine learning to analyze factors influencing match outcomes for close games in football (soccer). By focusing on specific features, we aim to uncover the key metrics that distinguish wins, losses, and draws in closely contested matches. The project involves data preprocessing, exploratory analysis, model training and evaluation, and feature importance interpretation.

# Project Overview
This project analyzes data from close football matches (goal difference of ±1 or 0) to:

### Identify key metrics (e.g., passes, pressures) that influence match outcomes.
### Build machine learning models to predict match outcomes.
### Interpret model results using SHAP values to gain insights into feature importance.

# Installation
To run this project, ensure you have the following libraries installed:

bash
Copy code
pip install pandas matplotlib seaborn scikit-learn shap

# Data Preprocessing
## Loading and Cleaning Data:
The dataset (CCFC_match_lineups_data.csv) is loaded, and missing values and duplicates are removed.
The lineup column is parsed to extract player-level data.

## Feature Engineering:
Calculated goal_difference as the difference between goals_scored and goals_conceded.
Filtered data to include only close games (goal difference ±1 or draws).
Extracted player details (player_id, squad_role, minutes_played) from the lineup data.

## Exploratory Analysis
Several visualizations were created to explore patterns in close games:

Win/Loss/Draw Distribution by Location: Examines the influence of game location on match outcomes.

Box Plots: Visualizes key match statistics by match outcome.

Correlation Heatmap: Shows relationships between key stats to identify interdependencies.

# Modeling
We used a Random Forest Classifier for each match outcome (win, loss, draw) with hyperparameter tuning to optimize performance:

## Hyperparameter Tuning: Conducted using GridSearchCV with cross-validation to find the best model parameters.
## Evaluation Metrics: Evaluated model performance using accuracy, cross-validation scores, classification reports, and confusion matrices.

# Feature Importance
Using SHAP (SHapley Additive exPlanations) values, we analyzed which features were most influential in determining each outcome:

For each outcome, the top 3 most important features are displayed, helping to interpret which metrics matter most in close games.

# Results
## Key Metrics for Each Outcome
Win: Features such as passes, pressures, and shots may be more significant in achieving a win.
Loss: Pressure regains and final third possession might play a more prominent role.
Draw: Defensive metrics such as tackles and shots conceded could influence draw outcomes.

## Key Takeaways
Location and Outcome Trends: Observed how location impacts win/loss likelihood in close games.
Feature Insights: Important features for each outcome help inform strategies for gameplay improvement.

## Future Work
Model Expansion: Experiment with other machine learning models (e.g., XGBoost, SVM) for potentially better performance.
Player-Specific Analysis: Focus on individual player contributions to the overall team performance.
Additional Data: Include more contextual data (weather, injuries) to refine the analysis.

# Acknowledgments
Special thanks to Coventry City Footbal for providing the match lineup dataset.
