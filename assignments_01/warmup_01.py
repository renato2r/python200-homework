# --- Pandas ---

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr


# Pandas Q1 - Create the following DataFrame and print the first three rows, the shape, and the data types of each column.

data = {
    "name": ["Alice", "Bob", "Charlie","None", "None"], #--Added None values
    "grade": [85, 72, 90, 68, 95],
    "city" : ["Boston", "Austin", "Boston", "Denver", "Austin" ],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)

print(f"Num Rows: {len(df.head(3))}\n")

print(f"Shape: {df.shape}\n")

print(f"Data Types:\n{df.dtypes}\n")

print("------------------------------------------------------------")

# Pandas Q2 - Using the DataFrame from Q1, filter the rows to show only students who passed and have a grade above 80. Print the result.

aprov = df[(df["passed"] == True) & (df["grade"] > 80)]
print(aprov)
print("------------------------------------------------------------")

# Pandas Q3 - Add a new column called "grade_curved" that adds 5 points to each student's grade. Print the updated DataFrame (all columns, all rows).
df["grade_curved"] = df["grade"] + 5
print(df)
print("------------------------------------------------------------")

# Pandas Q4 - Add a new column called "name_upper" that contains each student's name in uppercase, using the .str accessor. Print the "name" and "name_upper" columns together.
df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])
print("------------------------------------------------------------")

# Pandas Q5 - Group the DataFrame by "city" and compute the mean grade for each city. Print the result.
mean_by_city = df.groupby("city")["grade"].mean()
print(mean_by_city)
print("------------------------------------------------------------")

# Pandas Q6 - Replace the value "Austin" in the "city" column with "Houston". Print the "name" and "city" columns to confirm the change.
df = df.replace("Austin", "Houston")
print(df[["name", "city"]])
print("------------------------------------------------------------")

# Pandas Q7 - '
df = df.sort_values("grade", ascending=False)
print(df.head(3))
print("------------------------------------------------------------")

# NumPy Q1 - Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.

arr = np.array([10, 20, 30, 40, 50])
print(f"Shape: {arr.shape}\n")
print(f"Data Type: {arr.dtype}\n")
print(f"Number of Dimensions: {arr.ndim}\n")
print("------------------------------------------------------------")

# NumPy Q2 - Create the following 2D array and print its shape and size (total number of elements).
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(f"Shape: {arr.shape}\n")
print(f"Size: {arr.size}\n")
print("------------------------------------------------------------")

# NumPy Q3 - Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]].
sliced_arr = arr[:2, :2]
print(sliced_arr)
print("------------------------------------------------------------")

# NumPy Q4 - Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both.
zeros_arr = np.zeros((3, 4))
ones_arr = np.ones((2, 5))
print("3x4 Array of Zeros:\n", zeros_arr)
print("\n2x5 Array of Ones:\n", ones_arr)
print("------------------------------------------------------------")   

# NumPy Q5 - Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation.
arr = np.arange(0, 50, 5)
print(arr)
print(f"Shape: {arr.shape}\n")
print(f"Mean: {arr.mean()}\n")      
print(f"Sum: {arr.sum()}\n")
print(f"Standard Deviation: {arr.std()}\n")
print("------------------------------------------------------------")

# NumPy Q6 - Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result.
random_arr = np.random.normal(0, 1, 200)
print(f"Mean: {random_arr.mean()}\n")
print(f"Standard Deviation: {random_arr.std()}\n")
print("------------------------------------------------------------")

# Matplotlib Q1 - Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x, y)          
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y") 
plt.show()
print("------------------------------------------------------------")

# Matplotlib Q2 - Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.

subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores)    
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()
print("------------------------------------------------------------")

# Matplotlib Q3 - Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')
plt.title("Scatter Plot of Two Datasets")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
print("------------------------------------------------------------")

# Matplotlib Q4 - Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(x, y, color='blue')        
ax1.set_title("Squares")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax2.bar(subjects, scores, color='orange')
ax2.set_title("Subject Scores")
ax2.set_xlabel("Subjects")
ax2.set_ylabel("Scores")        
plt.tight_layout()
plt.show()  
print("------------------------------------------------------------")

# Descriptive Stats Q1 - Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard Deviation: {np.std(data)}")
print("------------------------------------------------------------")

# Descriptive Stats Q2 - Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.

# Generate data
data = np.random.normal(65, 10, 500)

# Create histogram
plt.hist(data, bins=20, color='skyblue', edgecolor='black')

# Add title and labels
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()
print("------------------------------------------------------------")

# Descriptive Stats Q3Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison".
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

data = [group_a, group_b]

plt.boxplot(data, labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()
print("------------------------------------------------------------")

# Descriptive Stats Q4 - You are given two datasets: one normally distributed and one 'exponential' distribution.

# data generation
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

# boxplots side by side
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])

plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.show()
print("------------------------------------------------------------")   

# Descriptive Stats Q5 - Print the mean, median, and mode of the following:

# Conjuntos de dados
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]


# Calculando e imprimindo os resultados
print(f"Data 1 - Mean: {np.mean(data1)}, Median: {np.median(data1)}")
print(f"Data 2 - Mean: {np.mean(data2)}, Median: {np.median(data2)}")

# The large discrepancy occurs due to the presence of an 'outlier' (extreme value): 150.

print("------------------------------------------------------------")  

# Hipothesis Q1 - Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")
print("------------------------------------------------------------")  

# Hipothesis Q2 - Using the p-value from Q1, write an if/else statement that prints whether the result is statistically significant at alpha = 0.05.
p_value = 1.5471178249432405e-06
alpha = 0.05

if p_value < alpha:
    print(f"The result is statistically significant (p = {p_value:.8f})")
    print("Decision: Reject the null hypothesis. There is a real difference between the groups.")
else:
    print(f"The result is NOT statistically significant (p = {p_value:.4f})")
    print("Decision: Fail to reject the null hypothesis. The difference may be due to chance")
print("------------------------------------------------------------")  
    
# Hipothesis Q3 - Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value.
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

# Data (same students - before and after)
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

# test t paired
t_stat, p_val = stats.ttest_rel(before, after)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")
print("------------------------------------------------------------")  

# Hipothesis Q4 - Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value.

# benchmark
scores = [72, 68, 75, 70, 69, 74, 71, 73]
national_benchmark = 70

# test T one sample
t_stat, p_val = stats.ttest_1samp(scores, national_benchmark)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_val}")
print("------------------------------------------------------------")  
# Hipothesis Q5 - Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. Print the resulting p-value. Use the alternative parameter.
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# test t independent unilateral (one-tailed)
t_stat, p_val = stats.ttest_ind(group_a, group_b, alternative='less')

print(f"T-statistic: {t_stat}")
print(f"P-value (one-tailed): {p_val}")
print("------------------------------------------------------------")  

# Hipothesis Q6 - Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a print() statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.
print("Conclusion: There is strong evidence that Group A scores are significantly lower than Group B scores. "
      "The probability of this difference being due to random chance is extremely low (less than 0.0001%), "
      "indicating a real and consistent performance gap between the two groups.")

print("------------------------------------------------------------")  

# Correlation Q1 - Compute the Pearson correlation between x and y below using np.corrcoef(). Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1]).

# Define two datasets with a non-linear relationship
x = [1, 2, 3, 4, 5]
y = [10, 8, 13, 15, 12]

# Compute the Pearson correlation matrix
# np.corrcoef returns a 2x2 matrix comparing all variables
matrix = np.corrcoef(x, y)

# Extract the specific correlation coefficient between x and y
# Position [0, 1] represents the correlation between the first and second list
correlation_xy = matrix[0, 1]

print(f"Correlation Coefficient: {correlation_xy:.4f}")

print("------------------------------------------------------------")

# Correlation Q2 - Use pearsonr() from scipy.stats to compute the correlation between x and y below. Print both the correlation coefficient and the p-value.


# Define the datasets
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]

# Compute the Pearson correlation coefficient and the p-value
# pearsonr returns: (correlation coefficient, two-tailed p-value)
correlation, p_value = pearsonr(x, y)

# Print the results to the console
print(f"Correlation Coefficient: {correlation}")
print(f"P-value: {p_value}")
print("------------------------------------------------------------")

# Correlation Q3 - Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.

# Define the dictionary with physical data
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28]
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(people)

# Compute the Pearson correlation matrix for all numeric columns
# The corr() method returns a correlation matrix of the DataFrame
correlation_matrix = df.corr()

# Print the resulting correlation matrix
print(correlation_matrix)
print("------------------------------------------------------------")   

# Correlation Q4 - Create a scatter plot of x and y below, which have a negative relationship. Add a title "Negative Correlation" and label both axes.

# Define the data points for x and y
# As x increases, y decreases, showing a negative relationship
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

# Create a scatter plot to visualize the correlation
plt.scatter(x, y)

# Set the plot title and axis labels
plt.title("Negative Correlation")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()
print("------------------------------------------------------------")

# Correlation Q5 - Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap".

# Create the DataFrame
df = pd.DataFrame(people)

# Calculate the Pearson correlation matrix
correlation_matrix = df.corr()

# Create a heatmap to visualize the correlation matrix
# annot=True displays the correlation values inside the cells
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Add a title to the heatmap
plt.title("Correlation Heatmap")
plt.show()

# Print the matrix for reference
print(correlation_matrix)
print("------------------------------------------------------------")

# Pipeline Q1

import numpy as np
import pandas as pd

# Input data with missing values
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    # Convert a NumPy array into a pandas Series named "values"
    return pd.Series(arr, name="values")

def clean_data(series):
    # Remove all NaN (Not a Number) values from the Series
    return series.dropna()

def summarize_data(series):
    # Calculate statistical metrics and return them as a dictionary
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0] # Access the first element of the mode Series
    }
    return summary

def data_pipeline(arr):
    # Chain the processing steps in a sequential pipeline
    # Step 1: Conversion
    raw_series = create_series(arr)
    
    # Step 2: Cleaning
    cleaned_series = clean_data(raw_series)
    
    # Step 3: Summarization
    results = summarize_data(cleaned_series)
    
    return results

# Execute the pipeline and store the result
analysis_results = data_pipeline(arr)

# Print each key and value from the summary dictionary
for key, value in analysis_results.items():
    print(f"{key.capitalize()}: {value:.2f}")

