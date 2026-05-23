#%%
from sklearn.datasets import load_iris
import pandas as pd

# Load the dataset object
iris = load_iris()

# Create the DataFrame with the features
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Add the 'species' column (
# We map the numbers (0, 1, 2) to the actual names (setosa, etc.)
df['species'] = [iris.target_names[i] for i in iris.target]

# function to calculate the interquartile range (IQR)
def get_IQR(x):
    return x.quantile(0.75) - x.quantile(0.25)

summary_table = df.groupby('species').agg(['mean', 'std', 'median', get_IQR])
print(summary_table.transpose())

#---------------------Histograms Visuals---------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean visual style
sns.set_theme(style="whitegrid")

# Create a figure with 4 subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# List of the 4 features to plot
features = iris.feature_names 

# Loop through features and axes to create the plots
for i, feature in enumerate(features):
    row = i // 2
    col = i % 2
    sns.histplot(data=df, x=feature, hue="species", kde=True, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()

#--------------------- Bootstrap Sampling ---------------------
#reference: Gemini's code from HW2, but I have rewritten it to be more efficient and easier to read. I also added comments to explain each step.
#prompt: "Write code to perform bootstrap sampling on the iris dataset to calculate confidence intervals for the mean and IQR of each feature for each species. Use 1000 iterations and calculate the 95% confidence intervals. Store the results in a DataFrame and print it."

import numpy as np
# Define our targets
species_list = df['species'].unique()
attributes = iris.feature_names
n_iterations = 1000
results = []

for spec in species_list:
    # Filter data for this species once
    spec_df = df[df['species'] == spec]
    
    for attr in attributes:
        # Extract the specific column data
        data = spec_df[attr]
        
        # Lists to store our bootstrap samples for each stat
        boot_means = []
        boot_iqrs = []
        
        for i in range(n_iterations):
            # 1. Resample WITH replacement
            sample = data.sample(n=len(data), replace=True)
            
            # 2. Calculate stats for this specific resample
            boot_means.append(sample.mean())
            # Use your get_IQR function here
            boot_iqrs.append(sample.quantile(0.75) - sample.quantile(0.25))
            
        # 3. Calculate the 95% Confidence Intervals (2.5% to 97.5%)
        mean_bounds = np.percentile(boot_means, [2.5, 97.5])
        iqr_bounds = np.percentile(boot_iqrs, [2.5, 97.5])
        
        # Store results in a dictionary to easily convert to a DataFrame later
        results.append({
            'Species': spec,
            'Attribute': attr,
            'Stat': 'Mean',
            'Lower': mean_bounds[0],
            'Upper': mean_bounds[1]
        })
        results.append({
            'Species': spec,
            'Attribute': attr,
            'Stat': 'IQR',
            'Lower': iqr_bounds[0],
            'Upper': iqr_bounds[1]
        })

# Turn the results into a final table
error_table = pd.DataFrame(results)
print(error_table.head(20))


# %%
