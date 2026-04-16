# --- scikit-learn API ---
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from pathlib import Path

# scikit-learn Question 1

# 1. Prepare the data
# X (years) must be 2D - the .reshape(-1, 1) handles this
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# 2. CREATE: Instantiate the model
model = LinearRegression()

# 3. FIT: Train the model (find the best line through the points)
model.fit(years, salary)

# 4. PREDICT: Estimate salaries for 4 and 8 years of experience
# We must pass the input in the same 2D format as the training data
years_to_predict = np.array([4, 8]).reshape(-1, 1)
predictions = model.predict(years_to_predict)

# 5. RESULTS: Extracting the learned parameters and predictions
print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Prediction for 4 years of experience: ${predictions[0]:,.2f}")
print(f"Prediction for 8 years of experience: ${predictions[1]:,.2f}")

#sklearn Question 2

# Start with a 1D array
x = np.array([10, 20, 30, 40, 50])
print(f"Original shape (1D): {x.shape}")

# Use .reshape() to convert it to a 2D array (5 rows, 1 column)
# The -1 means "calculate this dimension automatically"
x_2d = x.reshape(-1, 1)
print(f"New shape (2D): {x_2d.shape}")

# scikit-learn Question 3

# 1. Generate synthetic dataset
# X_clusters contains (x, y) coordinates for 120 points
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# 2. CREATE: Instantiate the KMeans model
# We define 3 clusters because we know the data was generated that way
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')

# 3. FIT: The model learns the positions of the 3 centers
kmeans.fit(X_clusters)

# 4. PREDICT: Assign a cluster label (0, 1, or 2) to each point
labels = kmeans.predict(X_clusters)

# 5. RESULTS: Print centers and counts
print("Cluster Centers (Coordinates):")
print(kmeans.cluster_centers_)

counts = np.bincount(labels)
for i, count in enumerate(counts):
    print(f"Cluster {i}: {count} points")

# 6. VISUALIZATION
plt.figure(figsize=(10, 6))

# Scatter plot of the points, colored by their labels
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', alpha=0.6, label='Data Points')

# Plot the centers as black X's
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, linewidths=3, label='Centroids')

plt.title("K-Means Clustering: Identifying 3 Natural Groups")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.legend()

# Save the figure (ensuring outputs directory exists)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUT_DIR / "kmeans_clusters.png")
plt.close()

print(f"\nVisual output saved to {OUTPUT_DIR / 'kmeans_clusters.png'}")

# --- Linear Regression ---

# Linear Regression Question 1 - 

# Re-using the data from the setup
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Visualizing the data
plt.figure(figsize=(10, 6))
# c=smoker uses the 0/1 values to pick colors from the coolwarm map
plt.scatter(age, cost, c=smoker, cmap="coolwarm", alpha=0.7)

plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Annual Medical Cost ($)")

# Ensure output directory exists
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/cost_vs_age.png')
plt.show()

# Comment:
# There are two very clear, distinct parallel lines of points. 
# The bottom group (non-smokers) and the top group (smokers). 
# This suggests that being a smoker adds a massive, constant "penalty" or offset 
# to the medical costs, regardless of age, which acts as a second intercept.

# Linear Regression Question 2

# Re-using the generated data
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# 1. Prepare X and y
# Even with one feature, X must be 2D
X = age.reshape(-1, 1)
y = cost

# 2. Split the data
# test_size=0.2 means 20% for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Print the shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

# Linear Regression Question 3

# 1. CREATE and FIT the model
model = LinearRegression()
model.fit(X_train, y_train)

# 2. PREDICT on the test set
y_pred = model.predict(X_test)

# 3. EVALUATE performance
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r_squared = model.score(X_test, y_test)

print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"RMSE (Root Mean Squared Error): ${rmse:.2f}")
print(f"R² (Coefficient of Determination): {r_squared:.4f}")

# Comment:
# The slope represents the average increase in medical costs for every 1-year 
# increase in age. In this case, for each additional year of age, the model 
# predicts an increase of approximately $XXX.XX in annual medical costs.

# Linear Regression Question 4

# 1. Prepare the full feature set
# np.column_stack combines age and smoker into a 2D array (100 rows, 2 columns)
X_full = np.column_stack([age, smoker])
y = cost

# 2. Split the data (using same random_state for consistency)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# 3. FIT the new model
model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)

# 4. EVALUATE and COMPARE
r2_full = model_full.score(X_test_f, y_test_f)

print(f"R² with age AND smoker: {r2_full:.4f}")
print(f"Age coefficient:      {model_full.coef_[0]:.2f}")
print(f"Smoker coefficient:   {model_full.coef_[1]:.2f}")

# Comment:
# The smoker coefficient represents the estimated average cost difference between 
# a smoker and a non-smoker, holding age constant. In practical terms, it's the 
# "price tag" of smoking: it tells us that being a smoker adds approximately 
# $15,000 to the annual medical bill compared to a non-smoker of the same age.

# Linear Regression Question 5

# 1. Setup Data (Recriando o dataset para garantir consistência)
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# 2. Train/Test Split (Idade + Fumante)
X_full = np.column_stack([age, smoker])
y = cost
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

# 3. Fit the model
model_full = LinearRegression()
model_full.fit(X_train, y_train)

# 4. Predict
y_pred = model_full.predict(X_test)

# 5. Visualization
plt.scatter(y_pred, y_test, alpha=0.7, color='blue', label='Patients')

# Diagonal reference line (y = x)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

plt.title("Predicted vs Actual Medical Costs")
plt.xlabel("Predicted Cost ($)")
plt.ylabel("Actual Cost ($)")
plt.legend()

# Save the figure
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/predicted_vs_actual.png')

# COMMENT:
# - A point ABOVE the diagonal line means the Actual cost was higher than the Predicted cost. 
#   In other words, the model UNDERESTIMATED the expenses for that patient.
# - A point BELOW the diagonal line means the Predicted cost was higher than the Actual cost. 
#   This indicates an OVERESTIMATION by the model.