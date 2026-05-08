# Task 1: Load and Explore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Garantir que a pasta de outputs exista
os.makedirs('outputs', exist_ok=True)

# 1. Load the dataset
# O separador ';' é crucial conforme vimos na inspeção inicial
df = pd.read_csv('student_performance_math.csv', sep=';')

# 2. Explore the structure
print(f"Dataset Shape: {df.shape}")
print("\nFirst five rows:")
print(df.head())
print("\nData types of all columns:")
print(df.dtypes)

# 3. Plot Histogram of G3
plt.figure(figsize=(10, 6))
# 21 bins cobrem exatamente a escala de 0 a 20
plt.hist(df['G3'], bins=21, range=(0, 20), color='skyblue', edgecolor='black')

plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

# 4. Save the figure
plt.savefig('outputs/g3_distribution.png')
plt.show()

print("\nHistogram saved to outputs/g3_distribution.png")

# Task 2: Preprocess the Data

# 1. Handle the G3=0 rows
# Reason: Keeping G3=0 distorted the model because these zeros likely represent 
# students who dropped out or missed the exam, not their actual math ability. 
# Linear regression tries to fit a line through all points; these outliers 
# would pull the slope down artificially.
df_filtered = df[df['G3'] > 0].copy()

print(f"Shape before filtering: {df.shape}")
print(f"Shape after filtering:  {df_filtered.shape}")
print(f"Rows removed: {df.shape[0] - df_filtered.shape[0]}")

# 2. Convert categorical columns to numeric
# Mapping yes/no and sex to 1/0
binary_mapping = {'yes': 1, 'no': 0}
sex_mapping = {'F': 0, 'M': 1}

cols_to_fix = ['schoolsup', 'internet', 'higher', 'activities']

for col in cols_to_fix:
    df_filtered[col] = df_filtered[col].map(binary_mapping)

df_filtered['sex'] = df_filtered['sex'].map(sex_mapping)

# 3. Correlation Analysis: Absences vs G3
corr_original = df['absences'].corr(df['G3'])
corr_filtered = df_filtered['absences'].corr(df_filtered['G3'])

print(f"\nCorrelation (Absences vs G3) - Original: {corr_original:.4f}")
print(f"Correlation (Absences vs G3) - Filtered: {corr_filtered:.4f}")

# Reasoning for the correlation shift:
# In the original data, many students with G3=0 actually had ZERO absences. 
# This created a contradiction: "perfect attendance = 0 grade," which 
# weakened the statistical relationship. Once filtered, we see the true 
# negative trend where more absences generally lead to lower grades.

# Task 3: Exploratory Data Analysis

# 1. Compute and sort correlations (excluding G1 and G2)
# Using only numeric/converted columns from the filtered dataset
correlations = df_filtered.drop(columns=['G1', 'G2']).corr()['G3'].sort_values()

print("Pearson Correlation with G3 (Sorted):")
print(correlations)

# The strongest negative relationship is 'failures' (-0.29).
# The strongest positive relationship is 'Medu' (0.19) and 'Fedu' (0.16).
# A surprising result is 'schoolsup' (school support) having a strong negative correlation (-0.24). 
# This is likely because students receiving extra support are those who were already struggling.

# 2. Visualizations
os.makedirs('outputs', exist_ok=True)

# Plot 1: Boxplot of failures vs G3
# Comment: This plot shows a clear negative trend. Students with zero past failures 
# have a much higher median grade and a tighter distribution than those with 1 or more.
plt.figure(figsize=(8, 6))
sns.boxplot(x='failures', y='G3', data=df_filtered, palette='Set2')
plt.title("Impact of Past Failures on Final Grade")
plt.xlabel("Number of Past Class Failures")
plt.ylabel("Final Grade (G3)")
plt.savefig('outputs/failures_vs_g3.png')

# Plot 2: Boxplot of Medu (Mother's Education) vs G3
# Comment: We can see a positive correlation here. As the mother's education level 
# increases from 0 to 4, the median grade of the students also tends to rise, 
# suggesting home environment plays a significant role in academic success.
plt.figure(figsize=(8, 6))
sns.boxplot(x='Medu', y='G3', data=df_filtered, palette='viridis')
plt.title("Mother's Education Level vs Final Grade")
plt.xlabel("Mother's Education Level (0-4)")
plt.ylabel("Final Grade (G3)")
plt.savefig('outputs/medu_vs_g3.png')

# Task 4: Baseline Model

# 1. Prepare features (X) and target (y)
# Using only 'failures' as the single feature
X_baseline = df_filtered[['failures']] 
y = df_filtered['G3']

# 2. Split into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42
)

# 3. Fit the Linear Regression model
model_baseline = LinearRegression()
model_baseline.fit(X_train, y_train)

# 4. Evaluation
y_pred = model_baseline.predict(X_test)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred))
r2_baseline = r2_score(y_test, y_pred)

print(f"--- Baseline Model (failures only) ---")
print(f"Slope: {model_baseline.coef_[0]:.2f}")
print(f"RMSE:  {rmse_baseline:.2f}")
print(f"R²:    {r2_baseline:.4f}")

# Comment:
# In plain English, the slope of -1.43 means that for every past class failure, 
# a student's final grade is expected to drop by about 1.43 points on the 0-20 scale. 
# The RMSE of 2.96 tells us that the model's predictions are, on average, off by 
# about 3 grade points. 
# The R² of ~0.09 is quite low, which is expected since we are ignoring many 
# other important factors like study time and family background that we saw in the EDA.

# Task 5: Build the Full Model

# 1. Define feature columns
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", 
                "internet", "sex", "freetime", "activities", "traveltime"]

# 2. Prepare X and y from the filtered dataframe
X = df_filtered[feature_cols].values
y = df_filtered["G3"].values

# 3. Split (80/20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Fit the model
model_full = LinearRegression()
model_full.fit(X_train, y_train)

# 5. Print results
train_r2 = model_full.score(X_train, y_train)
test_r2 = model_full.score(X_test, y_test)
y_pred = model_full.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Train R²: {train_r2:.4f}")
print(f"Test R²:  {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

print("\nFeatures and Coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# --- ANALYSIS COMMENTS ---

# 1. Surprising Results:
# The most surprising result is 'schoolsup' having a massive negative coefficient (-2.06). 
# Reasoning: As discussed in EDA, this is likely due to 'reverse causality' — students 
# don't perform poorly because of support; they are in support programs because 
# they were already failing. The model interprets 'schoolsup=1' as a red flag 
# for struggling students.

# 2. Train R² vs Test R²:
# Train R² (0.1749) and Test R² (0.1539) are very close. This gap is small, which 
# tells us the model is NOT overfitting. It generalizes reasonably well to 
# new data, although the overall predictive power remains low because human 
# behavior and grades are influenced by factors not captured in this dataset.

# 3. Comparison to Baseline:
# Adding more features improved the Test R² from ~0.09 (Task 4) to ~0.15. 
# This is a significant relative improvement (~66% better), showing that 
# variables like 'internet', 'higher', and 'studytime' add valuable context.

# 4. Deployment Recommendations:
# If deploying this model, I would KEEP features with high absolute coefficients 
# like 'failures', 'schoolsup', 'internet', and 'higher', as they are strong 
# indicators. I would DROP 'activities' (-0.009) and 'freetime' (-0.042) because 
# their coefficients are near zero, suggesting they have negligible impact 
# on predicting the final math grade in this specific model.


# Task 6: Evaluate and Summarize

# 1. Create Predicted vs Actual Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test, alpha=0.6, color='darkblue', label='Students')

# Add diagonal reference line (y = x)
line_coords = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(line_coords, line_coords, color='red', linestyle='--', label='Perfect Prediction')

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade (G3)")
plt.ylabel("Actual Grade (G3)")
plt.legend()

os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/predicted_vs_actual.png')

# COMMENT ON PLOT ANALYSIS:
# The model seems to struggle more at the high end (students with grades 18-20) 
# and the very low end. The error is not perfectly uniform; predictions are 
# clustered mostly between 8 and 14, showing the model is "conservative" and 
# avoids predicting extreme grades. 
# - A value ABOVE the diagonal means the Actual grade was higher than Predicted (Underestimation).
# - A value BELOW the diagonal means the Predicted grade was higher than Actual (Overestimation).

"""
--- MINI-PROJECT SUMMARY ---

1. DATASET SIZE:
   - Filtered dataset: 357 students (after removing 38 who scored G3=0).
   - Test set: 72 students (20% of the filtered data).

2. PERFORMANCE:
   - RMSE: 2.86. On a 0-20 scale, this means a typical prediction is off by 
     nearly 3 points. For a student actually scoring a 12, the model might 
     predict anything from a 9 to a 15.
   - Test R²: 0.1539. This means our model explains about 15% of the variance 
     in final math grades using only background and behavioral features.

3. KEY FEATURES:
   - Largest Negative: 'schoolsup' (-2.06) and 'failures' (-1.14). 
     This means that being in school support or having past failures are the 
     strongest predictors of lower final grades.
   - Largest Positive: 'internet' (+0.83) and 'higher' (+0.61). 
     This suggests that having internet access at home and the desire to 
     pursue higher education are the strongest positive indicators for math success.

4. SURPRISING RESULT:
   - The most surprising result was the strong negative impact of 'schoolsup'. 
     Intuitively, extra support should help, but the data reveals it acts as 
     a marker for students who are already significantly behind.
"""

# Neglected Feature: The Power of G1 ---

# 1. Update features to include G1
feature_cols_g1 = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", 
                   "internet", "sex", "freetime", "activities", "traveltime", "G1"]

X_g1 = df_filtered[feature_cols_g1].values
y = df_filtered["G3"].values

# 2. Split (using the same random_state for direct comparison)
X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1, y, test_size=0.2, random_state=42
)

# 3. Fit the model with G1
model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

# 4. Print the new result
test_r2_g1 = model_g1.score(X_test_g1, y_test_g1)
print(f"Test R² (with G1): {test_r2_g1:.4f}")

# --- REFLECTION COMMENTS ---

# 1. Correlation vs. Causality:
# Does a high R² mean G1 is causing G3? No. G1 and G3 are both measurements of 
# the same underlying variable: the student's mastery of the subject. G1 is 
# a "proxy" for academic ability. While it is a great predictor, it is a 
# correlate, not necessarily the root cause of the final grade.

# 2. Model Utility:
# Is this a useful model for identifying students who might struggle? 
# Only partially. By the time G1 is available, the student has already completed 
# a significant portion of the year. While it confirms who is struggling, it 
# might be "too late" for some interventions. It is more of a monitoring tool 
# than an early-warning system.

# 3. Early Intervention Strategy:
# What should educators do to intervene BEFORE G1? 
# They should rely on the model from Task 5. By looking at factors like past 
# failures, family education level, and home resources (internet), educators 
# can identify "at-risk" students on Day 1 of the semester, allowing for 
# preventive support before the first exam even happens.