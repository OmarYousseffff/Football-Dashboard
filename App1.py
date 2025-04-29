import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Configuration
st.set_page_config(
    page_title="Football Scouting: Bias-Variance Tradeoff",
    layout="centered",
)

# True price function (realistic football valuations)
def true_price(rating):
    base = 10 * (rating - 5)**1.8  # Base growth
    if rating > 7.5:
        base += 15 * (rating - 7.5)**3  # Star premium
    if rating > 8.5:
        base += 100 * (rating - 8.5)**4  # Superstar effect
    return np.clip(base, 0, 2000)  # Cap at €200M

# Data generation with adjustable noise
def generate_data(n_samples=100, noise_std=2):
    ratings = np.clip(np.random.normal(6.8, 0.9, n_samples), 5, 10)
    prices = np.array([true_price(r) + np.random.normal(0, noise_std*true_price(r)/10) for r in ratings])  # Fixed missing )
    return ratings.reshape(-1, 1), prices

# Core computation function
def compute_errors(degree, n_experiments=50, noise_std=2):
    x_test = np.linspace(5, 10, 100).reshape(-1, 1)
    y_true = np.array([true_price(x) for x in x_test.flatten()])
    predictions = []
    
    for _ in range(n_experiments):
        X, y = generate_data(noise_std=noise_std)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        predictions.append(model.predict(x_test))
    
    predictions = np.array(predictions)
    avg_pred = np.mean(predictions, axis=0)
    
    bias_squared = (avg_pred - y_true)**2
    variance = np.var(predictions, axis=0)  # Fixed typo
    noise = np.var([y - true_price(x) for x, y in zip(X.flatten(), y)])
    mse = bias_squared + variance + noise
    
    return x_test.flatten(), bias_squared, variance, noise, mse, avg_pred, predictions


# Main app
st.title("⚽ Football Player Valuation: Bias-Variance Tradeoff")

# Key explanation in sidebar
with st.sidebar:
    st.header("Key Concepts")
    st.markdown("""
    **MSE = Bias² + Variance + Noise**
    - **Bias²**: Systematically undervaluing stars  
    - **Variance**: Valuations change too much between scouts  
    - **Noise**: Unpredictable factors (injuries, contracts)  
    """)
    st.markdown("---")
    st.caption("Adjust the settings to see how model complexity affects prediction errors")

# Interactive controls
col1, col2 = st.columns(2)
with col1:
    degree = st.slider("Model Complexity (Polynomial Degree)", 1, 10, 1)
with col2:
    noise_level = st.slider("Market Volatility", 1, 5, 2)

# Compute metrics
x_vals, bias2, var, noise, mse, avg_pred, all_preds = compute_errors(degree)

# Main visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Model fits
X_train, y_train = generate_data(noise_std=noise_level)
ax1.scatter(X_train, y_train, alpha=0.5, label="Player Data")
ax1.plot(x_vals, [true_price(x) for x in x_vals], 'k-', label="True Value")
for i, pred in enumerate(all_preds[:10]):  # Show first 10 fits
    ax1.plot(x_vals, pred, color='blue', alpha=0.15)
ax1.plot(x_vals, avg_pred, 'r-', label="Average Prediction")
ax1.set_xlabel("Player Rating (1-10)")
ax1.set_ylabel("Market Value (€M)")
ax1.set_title("Model Predictions vs Reality")
ax1.legend()
ax1.grid(True)

# Right plot: Error decomposition
ax2.plot(x_vals, mse, label="Total Error (MSE)", linewidth=2)
ax2.plot(x_vals, bias2, '--', label="Bias²")
ax2.plot(x_vals, var, '--', label="Variance")
ax2.plot(x_vals, [noise]*len(x_vals), '--', label="Noise")
ax2.set_xlabel("Player Rating (1-10)")
ax2.set_ylabel("Error Components")
ax2.set_title("Error Decomposition")
ax2.legend()
ax2.grid(True)

st.pyplot(fig)

# Key insights box
if degree <= 2:
    st.warning("🔍 Underfitting: Simple models miss the superstar premium (high bias)")
elif degree >= 8:
    st.warning("🔍 Overfitting: Complex models are too sensitive to small changes (high variance)")
else:
    st.success("✅ Good balance: Captures trends without overreacting to noise")

# Mathematical summary
with st.expander("📊 See Detailed Calculations"):
    st.latex(r'''
        \text{MSE} = \mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2(\hat{f}(x)) + \text{Var}(\hat{f}(x)) + \sigma^2
    ''')
    st.write(f"""
    - Average Bias²: {np.mean(bias2):.1f}
    - Average Variance: {np.mean(var):.1f} 
    - Noise: {noise:.1f}
    - Total MSE: {np.mean(mse):.1f}
    """)

# Football examples
st.markdown("---")
st.subheader("Football Examples")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Rotation Player (6.5)**")
    st.write("- Bias dominates (linear models OK)")
with col2:
    st.markdown("**Star Player (7.8)**")
    st.write("- Need some complexity")
with col3:
    st.markdown("**Superstar (9.0)**")
    st.write("- High variance risk with complex models")