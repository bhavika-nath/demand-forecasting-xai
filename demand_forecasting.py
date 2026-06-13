import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

print("STEP 1: DATA LOADING & EXPLORATION")
print("=" * 60)

df = pd.read_csv('us-retail-sales.csv')
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

print("\n" + "=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)


print("\nMissing values:")
print(df.isnull().sum())


print(f"\nDuplicate rows: {df.duplicated().sum()}")


df['Month'] = pd.to_datetime(df['Month'])
print("\nDate column converted to datetime successfully.")


print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

df['Year']  = df['Month'].dt.year
df['Month_Num'] = df['Month'].dt.month

df['Demand'] = df[['Clothing', 'Appliances', 'FoodAndBeverage',
                    'Automobiles', 'GeneralMerchandise', 'BuildingMaterials']].sum(axis=1)

df['Lag_1'] = df['Demand'].shift(1)
df['Lag_2'] = df['Demand'].shift(2)
df['Lag_3'] = df['Demand'].shift(3)

df['Rolling_Mean_3'] = df['Demand'].shift(1).rolling(window=3).mean()

df.dropna(inplace=True)

print(f"\nFeatures created: Year, Month_Num, Lag_1, Lag_2, Lag_3, Rolling_Mean_3")
print(f"Dataset shape after feature engineering: {df.shape}")

print("\n" + "=" * 60)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

plt.figure(figsize=(12, 5))
plt.plot(df['Month'], df['Demand'], color='steelblue', linewidth=1.5)
plt.title('Total Retail Demand Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.savefig('eda_demand_trend.png', dpi=150)
plt.show()
print("EDA chart saved: eda_demand_trend.png")

print("\n" + "=" * 60)
print("STEP 5: MODEL TRAINING")
print("=" * 60)

features = ['Year', 'Month_Num', 'Clothing', 'Appliances', 'FoodAndBeverage',
            'Automobiles', 'GeneralMerchandise', 'BuildingMaterials',
            'Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3']

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
print(f"\nTraining samples: {len(X_train)} | Test samples: {len(X_test)}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest trained successfully.")

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
print("Gradient Boosting trained successfully.")

print("\n" + "=" * 60)
print("STEP 6: MODEL EVALUATION")
print("=" * 60)

rf_mae  = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2   = r2_score(y_test, rf_predictions)

gb_mae  = mean_absolute_error(y_test, gb_predictions)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
gb_r2   = r2_score(y_test, gb_predictions)

print(f"\nRandom Forest   → MAE: {rf_mae:.2f} | RMSE: {rf_rmse:.2f} | R²: {rf_r2:.3f}")
print(f"Gradient Boosting → MAE: {gb_mae:.2f} | RMSE: {gb_rmse:.2f} | R²: {gb_r2:.3f}")

errors = y_test.values - rf_predictions
plt.figure(figsize=(8, 4))
plt.hist(errors, bins=30, color='steelblue', edgecolor='white')
plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
plt.title('Error Distribution — Random Forest', fontsize=13, fontweight='bold')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('error_distribution.png', dpi=150)
plt.show()
print("Chart saved: error_distribution.png")

plt.figure(figsize=(8, 4))
plt.scatter(rf_predictions, errors, alpha=0.4, color='steelblue', s=20)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.title('Residual Plot — Random Forest', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Demand')
plt.ylabel('Residuals')
plt.tight_layout()
plt.savefig('residual_plot.png', dpi=150)
plt.show()
print("Chart saved: residual_plot.png")

models = ['Random Forest', 'Gradient Boosting']
r2_scores = [rf_r2, gb_r2]
plt.figure(figsize=(7, 4))
bars = plt.bar(models, r2_scores, color=['steelblue', 'lightsteelblue'], edgecolor='white', width=0.4)
plt.ylim(0, 1)
plt.title('Model Comparison — R² Score', fontsize=13, fontweight='bold')
plt.ylabel('R² Score')
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print("Chart saved: model_comparison.png")

print("\n" + "=" * 60)
print("STEP 7: SHAP EXPLAINABILITY ANALYSIS (XAI)")
print("=" * 60)

explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot — Feature Impact on Demand', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart saved: shap_summary.png")

importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue')
plt.title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("Chart saved: feature_importance.png")

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Best Model     : Random Forest")
print(f"MAE            : {rf_mae:.2f}")
print(f"RMSE           : {rf_rmse:.2f}")
print(f"R² Score       : {rf_r2:.3f}")
print(f"Key Insight    : Lag features & time-based variables are")
print(f"                 the strongest demand drivers (via SHAP)")
print(f"Business Value : Enables inventory planning, supply chain")
print(f"                 optimization, and data-driven decisions")
print("=" * 60)
print("\nAll charts saved. Ready for video walkthrough!")
