# --- Preprocessing ---

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Q1

 # Splitting the data into training and test sets (80/20)
# Stratify=y ensures that each class is represented proportionally in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Q1: Training and Test Shapes")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")

# Q2

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit only on X_train to learn the mean and standard deviation
# and transform both train and test sets to maintain consistency
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nQ2: Means of scaled training columns")
# Calculating means across rows (axis=0) for each feature
print(X_train_scaled.mean(axis=0))

# Comment: We fit the scaler on X_train only to prevent data leakage, ensuring 
# the model has no knowledge of the mean or distribution of the test set during training.

# --- KNN ---

# Q1

# Building a KNN classifier with k=5 using UNSCALED data
# This demonstrates why scaling is usually necessary for distance-based models
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)

y_pred_unscaled = knn_unscaled.predict(X_test)

print("\nKNN Q1: Accuracy and Report (Unscaled Data)")
print(f"Accuracy: {accuracy_score(y_test, y_pred_unscaled):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_unscaled))

# Q2

# Building a KNN classifier with k=5 using SCALED data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("\nKNN Q2: Accuracy (Scaled Data)")
print(f"Accuracy: {accuracy_score(y_test, y_pred_scaled):.4f}")

# Comment: For the Iris dataset, scaling often makes little to no difference because 
# all features are measured in the same units (cm) and have similar ranges. 
# However, scaling is crucial for KNN when features have different units or 
# vastly different scales (e.g., age vs. annual income).

# Q 3

# Evaluating the KNN (k=5) model using 5-fold cross-validation on unscaled data
knn_cv = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=5)

print("\nKNN Q3: 5-Fold Cross-Validation Scores (Unscaled)")
print(f"Individual fold scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# Comment: This result is more trustworthy than a single train/test split 
# because it evaluates the model's performance on multiple subsets of the data, 
# reducing the risk that the score was influenced by a lucky or unlucky random split.

# Q 4

# Testing different values of k to find the optimal hyperparameter
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

print("\nKNN Q4: Tuning k with Cross-Validation")
for k in k_values:
    knn_loop = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_loop, X_train, y_train, cv=5)
    print(f"k = {k:2d} | Mean CV Score: {scores.mean():.4f}")

# Comment: I would choose the k that yields the highest Mean CV Score while 
# remaining relatively low to avoid over-smoothing the decision boundaries. 
# Usually, k=5 or k=7 performs best on Iris, balancing bias and variance.

# --- Classifier Evaluation ---

# Q1
# Creating a confusion matrix using predictions from KNN Q1 (unscaled)
import os
cm = confusion_matrix(y_test, y_pred_unscaled)

# Displaying the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix: KNN (k=5) on Unscaled Data")

# Saving the output
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/knn_confusion_matrix.png')

# Comment: Looking at the matrix, the model most often confuses 'versicolor' 
# and 'virginica'. This is expected as these two species have overlapping 
# petal measurements, unlike 'setosa' which is very distinct and easily separated.

# --- Decision Trees ---

# Q1

# Building a Decision Tree Classifier with a depth limit to prevent overfitting
dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_clf.fit(X_train, y_train)

y_pred_dt = dt_clf.predict(X_test)

print("\nDecision Tree Q1: Accuracy and Report")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# Comment 1: The Decision Tree accuracy is typically very similar to KNN on the 
# Iris dataset. While KNN looks at local neighbors, the Decision Tree finds 
# global axis-aligned splits that separate the species just as effectively here.

# Comment 2: Scaling would make no difference for a Decision Tree. Since trees 
# split data based on thresholds (e.g., "is petal width > 0.8?"), the relative 
# order of values matters, but their absolute scale or distance from each other does not.

# --- Logistic Regression and Regularization ---

# Q1

# Comparing models with different regularization strengths (C parameter)
c_values = [0.01, 1.0, 100]

print("\nLogistic Regression Q1: Impact of C on Coefficients")
for c in c_values:
    # Trocamos 'liblinear' por 'lbfgs' para suportar as 3 classes do Iris
    log_reg = LogisticRegression(C=c, max_iter=1000, solver='lbfgs')
    log_reg.fit(X_train_scaled, y_train)
    
    coef_sum = np.abs(log_reg.coef_).sum()
    print(f"C = {c:6.2f} | Total Coefficient Magnitude: {coef_sum:.4f}")

# Comment: As C increases, the total coefficient magnitude also increases. 
# This tells us that regularization is stronger when C is small (penalty is high), 
# forcing coefficients toward zero to prevent overfitting; as C grows, 
# the model is allowed to fit the data more aggressively with larger coefficients.

# --- PCA ---

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# Q1
# Print the shapes to understand the feature space (64D) vs spatial structure (8x8)
print(f"X_digits shape: {X_digits.shape}") # Flattened version for ML
print(f"images shape:   {images.shape}")   # 2D version for plotting

# Create a 1-row subplot showing one example of each digit class (0-9)
fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for i in range(10):
    # Find the first index where target matches the digit i
    idx = np.where(y_digits == i)[0][0]
    
    # Use gray_r (reversed grayscale) for dark ink on light background
    axes[i].imshow(images[idx], cmap='gray_r')
    axes[i].set_title(f"Digit: {i}")
    axes[i].axis('off')

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/sample_digits.png')

# Comment: This visualization confirms the data structure. Each digit is represented 
# by an 8x8 grid of pixels. The PCA's job will be to project these 64 individual 
# pixel values into a lower-dimensional space while preserving as much info as possible.

# 2
# Fitting PCA without specifying n_components defaults to min(n_samples, n_features)
pca = PCA()
pca.fit(X_digits)

# Transforming the data to get the scores (loadings for each sample)
scores = pca.transform(X_digits)

# Plotting the first two principal components
plt.figure(figsize=(10, 8))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)
plt.colorbar(scatter, label='Digit')

plt.title("Digits Dataset: PCA 2D Projection (PC1 vs PC2)")
plt.xlabel("Principal Component 1 Score")
plt.ylabel("Principal Component 2 Score")

# Saving the figure
plt.savefig('outputs/pca_2d_projection.png')

# Comment: Yes, same-digit images tend to cluster together in this 2D space. 
# While there is some overlap, clear groups are visible for digits like 0, 4, and 6, 
# showing that PC1 and PC2 capture enough variance to separate the classes significantly.

# Q3
# Calculating the cumulative sum of explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')

# Adding a horizontal line at 80% for reference
plt.axhline(y=0.8, color='r', linestyle=':', label='80% Explained Variance')

plt.title("Cumulative Explained Variance by Number of Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.legend()

# Saving the figure
plt.savefig('outputs/pca_variance_explained.png')

# Comment: To explain 80% of the variance, we need approximately 13 to 15 components. 
# This shows the power of PCA, as we can reduce the data from 64 dimensions to 
# less than a quarter of that size while still retaining the vast majority 
# of the original information.

# Q4

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# Settings for the reconstruction grid
n_values = [2, 5, 15, 40]
n_digits = 5

# Create a grid: Original row + 4 reconstruction rows = 5 rows total
fig, axes = plt.subplots(len(n_values) + 1, n_digits, figsize=(12, 10))

# Row 0: Original Images
for i in range(n_digits):
    axes[0, i].imshow(images[i], cmap='gray_r')
    axes[0, i].set_title(f"Original (Digit {y_digits[i]})")
    axes[0, i].axis('off')

# Rows 1-4: Reconstructions for different n_components
for row_idx, n in enumerate(n_values):
    for col_idx in range(n_digits):
        reconstructed = reconstruct_digit(col_idx, scores, pca, n)
        axes[row_idx + 1, col_idx].imshow(reconstructed, cmap='gray_r')
        axes[row_idx + 1, col_idx].set_title(f"n = {n}")
        axes[row_idx + 1, col_idx].axis('off')

plt.tight_layout()
plt.savefig('outputs/pca_reconstructions.png')

# Comment: The digits typically become clearly recognizable around n=15. 
# This matches the variance curve from Question 3, as we previously saw that 
# ~15 components capture 80% of the variance, which is where the 'elbow' 
# starts to level off and the reconstruction gains enough detail for human reading.