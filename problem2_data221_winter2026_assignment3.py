import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_csv('crime1.csv')
data = df['ViolentCrimesPerPop']

# create histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Violent Crimes Per Population')
plt.xlabel('Violent Crimes Per Population')
plt.ylabel('Frequency')
plt.show()

# create box plot
plt.boxplot(data, vert=False, patch_artist=True)
plt.title('Box Plot of Violent Crimes Per Population')
plt.xlabel('Violent Crimes Per Population')
plt.yticks([1], ['Crime Data']) # Simplifies the Y-axis label
plt.show()

# --- ANALYSIS COMMENTS ---
# The histogram shows how the data values are spread by illustrating the frequency of specific ranges.
# In this dataset, the histogram typically reveals a right-skewed distribution where most values
# cluster at the lower end with a long tail extending toward the higher values.
# The box plot represents the median as a vertical line inside the central box, indicating
# the middle value of the dataset. Regarding outliers, the box plot explicitly highlights them
# as individual points or markers located beyond the 'whiskers.' Their presence suggests that
# certain communities have significantly higher crime rates than the rest of the population.