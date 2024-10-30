import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import shap
import numpy as np
import math
import ast 

# Load the dataset
df = pd.read_csv('CCFC_match_lineups_data.csv')

# Drop rows with missing values and duplicates
df = df.dropna().drop_duplicates()
# Parse 'lineup' column to list of dictionaries if it contains JSON-like data for players
df['lineup'] = df['lineup'].apply(lambda x: ast.literal_eval(x) if x != '[]' else [])

# Create goal difference column
df['goal_difference'] = df['goals_scored'] - df['goals_conceded']

# Filter for close games (goal difference ±1 or draws)
close_games = df[(abs(df['goal_difference']) == 1) | (df['goal_difference'] == 0)]
# Extract 'player_id' and other details from the 'lineup' data
def extract_player_data(lineup_data):
    players = []
    for player in lineup_data:
        player_id = player.get('player_id')
        squad_role = player.get('squad_role')
        minutes_played = player.get('minutes_played', 0)  # Include minutes_played if available
        players.append({'player_id': player_id, 'squad_role': squad_role, 'minutes_played': minutes_played})
    return players

# Apply the function to extract player information
df['players'] = df['lineup'].apply(extract_player_data)
players_df = df[['match_id', 'match_outcome', 'goal_difference', 'players']].explode('players').reset_index(drop=True)
players_df = pd.concat([players_df.drop(['players'], axis=1), players_df['players'].apply(pd.Series)], axis=1)

# Verify 'player_id' exists in the exploded DataFrame
print("Columns in players_df:", players_df.columns)
# Define key stats (features) for analysis
key_stats = [
    'shots', 'passes', 'pressures', 'pressure_regains', 'shots_on_target', 
    'completed_passes_into_the_box', 'tackles', 'final_third_possession', 'ppda',
    'completed_passes_and_carries_into_final_third', 'shots_conceded_within_8_seconds_of_corner'
]
# Exploratory Analysis
# 1. Count Plot: Win/Loss/Draw distribution by location for close games
plt.figure(figsize=(10, 8))
sns.countplot(data=close_games, x='match_outcome', hue='location', palette='coolwarm')
plt.title('Win/Loss/Draw Rate by Location for Close Games\nHighlighting Outcome Trends by Location')
plt.xlabel('Match Outcome')
plt.ylabel('Count')
plt.show()

# 2. Box Plots for key stats by match outcome
num_plots = len(key_stats)
num_cols = 3
num_rows = math.ceil(num_plots / num_cols)

plt.figure(figsize=(20, 7 * num_rows))
for i, stat in enumerate(key_stats, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(data=close_games, x='match_outcome', y=stat, palette='viridis')
    plt.title(f'{stat.capitalize()} by Match Outcome')
    plt.xlabel('Match Outcome')
plt.tight_layout()
plt.show()

# 3. Correlation Heatmap to observe relationships between key stats
plt.figure(figsize=(10, 8))
correlation_matrix = close_games[key_stats].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Key Stats in Close Games\nObserving Feature Interdependencies')
plt.show()
# Reduced parameter grid for faster training
param_grid = {
    'n_estimators': [50, 100],        # Fewer estimators
    'max_depth': [5, 10],             # Limited depth options
    'min_samples_split': [5],         # Fixed min_samples_split
    'min_samples_leaf': [2]           # Fixed min_samples_leaf
}
# List to store top 3 features for each outcome
top_features = {'win': [], 'loss': [], 'draw': []}
# Function for model training, evaluation, and plotting
def train_evaluate_model(data, target_value, title_suffix):
    X = data[key_stats]
    y = data['match_outcome'].apply(lambda x: 1 if x == target_value else 0)  # Binary target: 1 for target_value, 0 otherwise

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning with GridSearchCV for Random Forest (faster setup)
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Best model from GridSearchCV
    best_model = grid_search.best_estimator_

    # Cross-validation score for the best model
    cv_scores = cross_val_score(best_model, X, y, cv=3)  # Reduced to 3 folds for speed
    print(f"Cross-Validation Scores ({title_suffix}):", cv_scores)
    print(f"Average CV Score ({title_suffix}):", np.mean(cv_scores))

    # Evaluate the tuned model on the test set
    y_pred = best_model.predict(X_test)
    print(f"\nOptimized Model Accuracy ({title_suffix}):", accuracy_score(y_test, y_pred))
    print(f"\nClassification Report ({title_suffix}):\n", classification_report(y_test, y_pred))

    # Get feature importances and select top 3
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': key_stats, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    top_3_features = feature_importance_df.head(3)['Feature'].tolist()
    top_features[target_value] = top_3_features  # Store top 3 features in dictionary

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'Feature Importances in Optimized Random Forest Model\n{title_suffix}')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

    # Plotting the classification report as a heatmap
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :].T  # Exclude the 'accuracy' row

    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df, annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title(f'Classification Report Heatmap\nEvaluating Model Performance on {title_suffix}')
    plt.show()

    # Confusion matrix plot for the optimized model
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, display_labels=['Other', target_value], cmap='Blues')
    plt.title(f'Confusion Matrix for Optimized Model\n{title_suffix}')
    plt.show()

    # SHAP Interpretation - selecting the SHAP values for the positive class only
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_train)
    
    # Select only the SHAP values for the positive class (index 1)
    shap_values_positive_class = shap_values.values[:, :, 1]
    
    # Plot SHAP summary plot for the selected class
    shap.summary_plot(shap_values_positive_class, features=X_train, feature_names=key_stats, plot_type="bar")

# Models for Wins, Losses, and Draws
# Filter datasets for each outcome
close_games_win = close_games[close_games['match_outcome'] == 'won']
close_games_loss = close_games[close_games['match_outcome'] == 'lost']
close_games_draw = close_games[df['match_outcome'] == 'draw']

# Train and evaluate models, capturing top features for each outcome
if close_games_win.shape[0] > 0:
    win_model = train_evaluate_model(close_games, 'won', 'Close Games Ending in Win')

if close_games_loss.shape[0] > 0:
    loss_model = train_evaluate_model(close_games, 'lost', 'Close Games Ending in Loss')

if close_games_draw.shape[0] > 0:
    draw_model = train_evaluate_model(close_games, 'draw', 'Close Games Ending in Draw')

# Define the categories to display based on non-empty features
non_empty_outcomes = {outcome: features for outcome, features in top_features.items() if features}

# Plot for each non-empty outcome
for outcome, features in non_empty_outcomes.items():
    # Simulate some example average values for demonstration (replace with real values if available)
    avg_values = [10, 20, 15]  # Replace with actual averages for each feature if available

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.bar(features, avg_values)
    plt.title(f"Top 3 Features for Matches Ending in {outcome.capitalize()}")
    plt.xlabel("Features")
    plt.ylabel("Average Value")
    plt.show()
    
# Goodbye message plot
plt.figure(figsize=(10, 6))
summary_features = ['passes', 'pressures']
focus_levels = [3, 5]  # Example focus levels for emphasis, where higher means more focus
colors = ['green', 'red']

plt.bar(summary_features, focus_levels, color=colors)
plt.title("Key Focus Areas Based on Match Outcomes")
plt.xlabel("Key Metrics")
plt.ylabel("Focus Level")
plt.ylim(0, 6)
plt.text(-0.15, 3.2, "Focus on 'passes' when winning", fontsize=12, color='green', weight='bold')
plt.text(0.85, 5.2, "Focus on 'pressures' when losing or drawing", fontsize=12, color='red', weight='bold')
plt.show()
