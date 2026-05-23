#
import pandas as pd
import numpy as np

# Load the dataset, Go to my current working folder and load this file.”
df = pd.read_excel('hw3.data1.xlsx')

print(df.head()) # print the first few rows of the dataser
print(df.columns.to_list()) # list of column names

print(df.shape) # number of rows and columns
print(df.dtypes) # data types of each column

print("The value count in column 'label':")
print(df['label'].value_counts()) #count # of times each value appears

print("The Prediction using the formula:")
weights = [24, -15, -38, -7, -41, 35, 0, -2, 19, 33, -3, 7, 3, -47, 26, 10, 40, -1, 3, 0]  # All 20 weights from the formula
bias = -6   #The bias term is the # without a feature
feature_cols = [f'column{i}' for i in range(1, 21)]

y_raw = df[feature_cols].dot(weights) + bias # calculate the prediction values using the dot product of features and weights, plus the bias
df['prediction'] = y_raw.apply(lambda x: 1 if x > 0 else -1) # apply the threshold function to convert raw prediction values to binary labels (1 or -1)


print(df[['label', 'prediction']].head()) # print the true labels & predicted labels for the first few rows

value_counts = df['prediction'].value_counts()
print("The Total value count in column 'prediction':")
print(value_counts) 

print("The Accuracy Matrix:")
df['correct'] = df['prediction'] == df['label']
accuracy = df['correct'].mean()
print(f"Accuracy: {accuracy:.2f}")

print ("The Confusion Matrix:")
TP = ((df['prediction'] == 1) & (df['label'] == 1)).sum() # True Positives
TN = ((df['prediction'] == -1) & (df['label'] == -1)).sum() # True Negatives
FP = ((df['prediction'] == 1) & (df['label'] == -1)).sum() # False Positives
FN = ((df['prediction'] == -1) & (df['label'] == 1)).sum()  # False Negatives
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}") # Check the counts to see if it adds up to 10000

print("The economic Calculation):Based on False Positives and False Negatives:")
economic_calc = (FN * 1000) + (FP * 100) 
print(f"Economic Loss: ${economic_calc}") # its a economic loss

fn_cost = FN * 1000
fp_cost = FP * 100
print ("Breakdown of Economic Loss: By False Negatives and False Positives:")

print(f"Proportion of loss from false negatives: {fn_cost / economic_calc * 100:.2f}%")
print(f"Proportion of loss from false positives: {fp_cost / economic_calc * 100:.2f}%")

#observation suggest the we have more false negatic which is more costly than false positives, so we should focus on reducing false negatives to minimize economic loss.

results = [] 
for t in range(-2000, 500, 10): # this thresshold can be adjusted to see how it affects the economic loss
    predictions = (y_raw > t).apply(lambda x: 1 if x else -1)
    FN_t = ((df['label'] == 1) & (predictions == -1)).sum()
    FP_t = ((df['label'] == -1) & (predictions == 1)).sum()
    loss = (FN_t * 1000) + (FP_t * 100)
    results.append({'threshold': t, 'loss': loss, 'FN': FN_t, 'FP': FP_t}) 

results_df = pd.DataFrame(results)
best = results_df.loc[results_df['loss'].idxmin()]
print(best)

# My current threshold found the FN to be at 0 which is a signicant reduction in economic loss
print("Top 10 Thresholds with the Lowest Economic Loss:")
print(results_df.sort_values('loss').head(10))

# Model that maximizes economic gain -> is a threshold of -1010 which brings our loss from 755400 to 112,400. 