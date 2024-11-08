from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# For Decision Tree Visualization
import matplotlib.pyplot as plt

# For Random Forest Tree Visualization
from sklearn import tree

# Load data
df = pd.read_csv('data/traffic_data.csv')

# Separate features and target
X = df.drop(columns=['Delay', 'Route'])  # Exclude target variable 'Delay'
y = df['Delay']

# Define categorical and numerical columns
categorical_cols = ['TrafficCondition', 'RoadCondition', 'WeatherCondition', 'TimeOfDay', 'DayType']
numerical_cols = ['BusCapacity', 'AvgSpeed', 'DistanceToDestination']

# Preprocessing pipeline (Scaling and OneHotEncoding)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Scale numerical features
        ('cat', OneHotEncoder(), categorical_cols)  # Encode categorical features
    ])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Feature Selection using SelectKBest (Choose top k features)
k = 8  # Number of best features to select
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X_preprocessed, y)

# Save the preprocessor and selector for future use
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

with open('models/feature_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
def tune_model(model, param_distributions, X_train, y_train):
    random_search = RandomizedSearchCV(
        model, 
        param_distributions, 
        n_iter=10, 
        scoring='neg_mean_squared_error', 
        n_jobs=-1, 
        cv=3, 
        random_state=42
    )
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

# Model training and evaluation with visualization
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'Decision Tree': DecisionTreeRegressor()
    }

    # Hyperparameter tuning configurations
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    dt_params = {
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        if name == 'Random Forest':
            model = tune_model(model, rf_params, X_train, y_train)
        elif name == 'Decision Tree':
            model = tune_model(model, dt_params, X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # Save the trained model
        trained_models[name] = model

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append({
            'Model': name,
            'MSE': mse,
            'MAE': mae,
            'R²': r2
        })
        print(f"{name} - MSE: {mse:.2f}, R²: {r2:.2f}, MAE: {mae:.2f}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # ====== Added Visualizations ======

    # 1. Linear Regression: Actual vs Predicted Scatter Plot
    lr_model = trained_models['Linear Regression']
    y_pred_lr = lr_model.predict(X_test)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Linear Regression: Actual vs Predicted Delay')
    plt.xlabel('Actual Delay')
    plt.ylabel('Predicted Delay')
    plt.show()

    # 2. Decision Tree: Plot the Tree Structure
    dt_model = trained_models['Decision Tree']
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_model, 
        feature_names=get_selected_feature_names(preprocessor, selector),
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title('Decision Tree Structure')
    plt.show()

    # 3. Random Forest: Plot a Single Tree from the Forest
    rf_model = trained_models['Random Forest']
    plt.figure(figsize=(20, 10))
    # Select the first tree in the forest
    tree_to_plot = rf_model.estimators_[0]
    plot_tree(
        tree_to_plot, 
        feature_names=get_selected_feature_names(preprocessor, selector),
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title('Random Forest: Single Tree Structure')
    plt.show()

    # ====== End of Added Visualizations ======

    return trained_models, results_df

# Helper function to get selected feature names after preprocessing and feature selection
def get_selected_feature_names(preprocessor, selector):
    # Get feature names from preprocessor
    categorical_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(['TrafficCondition', 'RoadCondition', 'WeatherCondition', 'TimeOfDay', 'DayType'])
    numerical_cols = ['BusCapacity', 'AvgSpeed', 'DistanceToDestination']
    all_feature_names = np.concatenate([numerical_cols, categorical_cols])

    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = all_feature_names[selected_indices]
    return selected_feature_names

# Train models and visualize
trained_models, results_df = train_models(X_train, X_test, y_train, y_test)

# Save all trained models
for name, model in trained_models.items():
    filename = f'models/{name.replace(" ", "_").lower()}_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

print("All models saved successfully!")

# Optionally, save the results dataframe for later analysis
results_df.to_csv('models/model_performance.csv', index=False)
print("Model performance metrics saved successfully!")
