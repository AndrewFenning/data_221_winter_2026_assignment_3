import pandas as pd

df = pd.read_csv("crime1.csv")

target_col = df['ViolentCrimesPerPop']

#compute statistical measures
mean_val = target_col.mean()
median_val = target_col.median()
std_val = target_col.std()
min_val = target_col.min()
max_val = target_col.max()

#print results
print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Standard Deviation: {std_val}")
print(f"Minimum Value: {min_val}")
print(f"Maximum Value: {max_val}")

# --- ANALYSIS COMMENTS ---

# Q: Compare the mean and median. Does the distribution look symmetric or skewed?
# A: If the mean is significantly greater than the median, the distribution is "Right-Skewed" (Positively Skewed).
#    If they are roughly equal, it is symmetric. In crime1.csv, we see a right skew because mean > median (significantly)

# Q: If there are extreme values, which statistic is more affected: mean or median?
# A: The mean is more affected. The mean is "sensitive" to outliers because it sums all values;
#    one extremely high value will pull the average up. The median is "robust" because it
#    only cares about the middle position of the data, regardless of how large the extremes are.