"""
NBA Player Salary Prediction: Horizontal Line + 4th-degree Polynomial Max Model
Best R² = 0.6118
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================================
# 1. Data Loading
# ============================================================================
df = pd.read_csv('csv_fold/merged_player_data.csv')
df_clean = df[['athletic_score', 'salary']].dropna()
X = df_clean['athletic_score'].values
y = df_clean['salary'].values

# ============================================================================
# 2. Model Parameters
# ============================================================================
SPLIT_POINT = 0.3888  # Optimal split point
DEGREE = 4            # Polynomial degree

# ============================================================================
# 3. Data Segmentation
# ============================================================================
mask_high = X >= SPLIT_POINT  # High segment mask

# Low segment data (for horizontal line)
X_low = X[~mask_high]
y_low = y[~mask_high]

# High segment data (for 4th-degree polynomial)
X_high = X[mask_high].reshape(-1, 1)
y_high = y[mask_high]

# ============================================================================
# 4. Model Fitting
# ============================================================================

# 4.1 Low segment: Horizontal line (mean value)
y_horizontal = y_low.mean()

# 4.2 High segment: 4th-degree polynomial regression
poly = PolynomialFeatures(degree=DEGREE)
X_high_poly = poly.fit_transform(X_high)
model_poly = LinearRegression()
model_poly.fit(X_high_poly, y_high)

# Get polynomial coefficients
coefficients = model_poly.coef_
intercept = model_poly.intercept_

# ============================================================================
# 5. Prediction Function
# ============================================================================

def predict_salary(athletic_score):
    """
    Salary prediction function
    
    Parameters:
        athletic_score: Athletic ability score (scalar or array)
    
    Returns:
        Predicted salary
    """
    x = np.atleast_1d(athletic_score)
    
    # Horizontal line prediction
    y_flat = np.full(len(x), y_horizontal)
    
    # 4th-degree polynomial prediction
    x_reshaped = x.reshape(-1, 1)
    x_poly = poly.transform(x_reshaped)
    y_curve = model_poly.predict(x_poly)
    
    # Max continuity
    y_pred = np.maximum(y_flat, y_curve)
    
    if len(y_pred) == 1:
        return y_pred[0]
    return y_pred

# ============================================================================
# 6. Model Evaluation
# ============================================================================

# Predict all data
y_pred = predict_salary(X)

# Calculate metrics
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = np.mean(np.abs(y - y_pred))
residual_variance = np.var(y - y_pred)

print("=" * 80)
print("MODEL PARAMETERS")
print("=" * 80)
print(f"Split point: {SPLIT_POINT}")
print(f"Horizontal line value: ${y_horizontal:,.2f}")
print(f"\n4th-degree polynomial coefficients:")
print(f"  Intercept (constant):  {intercept:>20,.2f}")
print(f"  x^1 coefficient:       {coefficients[1]:>20,.2f}")
print(f"  x^2 coefficient:       {coefficients[2]:>20,.2f}")
print(f"  x^3 coefficient:       {coefficients[3]:>20,.2f}")
print(f"  x^4 coefficient:       {coefficients[4]:>20,.2f}")
print("=" * 80)

print("\nMODEL PERFORMANCE")
print("=" * 80)
print(f"R² (Coefficient of Determination):  {r2:.6f}")
print(f"RMSE (Root Mean Squared Error):     ${rmse:,.2f}")
print(f"MAE (Mean Absolute Error):          ${mae:,.2f}")
print(f"Residual Variance:                  {residual_variance:,.2f}")
print(f"Sample size:                        {len(X)}")
print(f"  Low segment samples:              {np.sum(~mask_high)}")
print(f"  High segment samples:             {np.sum(mask_high)}")
print("=" * 80)

# ============================================================================
# 7. Model Expression
# ============================================================================
print("\nMODEL EXPRESSION")
print("=" * 80)
print(f"For x < {SPLIT_POINT:.4f}:")
print(f"  y = {y_horizontal:,.2f}")
print(f"\nFor x >= {SPLIT_POINT:.4f}:")
print(f"  p(x) = {intercept:,.2f}")
print(f"         + {coefficients[1]:,.2f} * x")
print(f"         + {coefficients[2]:,.2f} * x^2")
print(f"         + {coefficients[3]:,.2f} * x^3")
print(f"         + {coefficients[4]:,.2f} * x^4")
print(f"\nMax Model:")
print(f"  y = max({y_horizontal:,.2f}, p(x))")
print("=" * 80)

# ============================================================================
# 8. Visualization (Single Fitting Plot)
# ============================================================================

# Create smooth prediction curve
X_plot = np.linspace(X.min(), 0.8, 1000)
y_plot_horizontal = np.full(len(X_plot), y_horizontal)
y_plot_curve = model_poly.predict(poly.transform(X_plot.reshape(-1, 1)))
y_plot_final = np.maximum(y_plot_horizontal, y_plot_curve)

# Create figure with size 7.4 x 5.4 inches
fig, ax = plt.subplots(figsize=(7.4, 4))

# Scatter plot with different colors for low and high segments
ax.scatter(X[~mask_high], y[~mask_high], alpha=0.5, s=40, 
           color='lightblue', edgecolors='blue', linewidths=0.5,
           label=f'Low segment (x < {SPLIT_POINT:.3f})')
ax.scatter(X[mask_high], y[mask_high], alpha=0.5, s=40, 
           color='lightcoral', edgecolors='red', linewidths=0.5,
           label=f'High segment (x ≥ {SPLIT_POINT:.3f})')

# Fitted curve
ax.plot(X_plot, y_plot_final, 'red', linewidth=3, 
        label=f'Max fitted curve (R² = {r2:.4f})', zorder=10)

# Split point vertical line
ax.axvline(x=SPLIT_POINT, color='green', linestyle='--', linewidth=2,
           label=f'Split point: {SPLIT_POINT:.4f}')

# Labels and formatting
ax.set_xlabel('Athletic Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
ax.set_title('NBA Player Salary Prediction: Max Model Fit', 
             fontsize=14, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nba_salary_max_model_fit.pdf', dpi=300, bbox_inches='tight')
print("\n✓ Figure saved: nba_salary_max_model_fit.png")

plt.show()

print("\nCOMPLETED!")
