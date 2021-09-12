###############################################################
# Main function: Analyze which factors have a greater impact on the total price, and display it visually
###############################################################
import warnings
warnings.filterwarnings("ignore")
from preprocess import data_preprocess
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read in the original data and calculate the correlation coefficient
data = pd.read_csv('data/green_tripdata_2020-04.csv')
data = data.drop(['ehail_fee'], axis=1)
corr = data.corr()
corr.to_csv('corr.csv')

# Visualization correlation coefficient
plt.figure(frameon = True)
ax = sns.heatmap(corr)
plt.tight_layout(0.8)
plt.savefig('corr.png')
plt.show()

# Select several features with larger correlation coefficients and draw a scatter plot
# Here we select the data from April for visualization
X = data_preprocess('data/green_tripdata_2020-04.csv', 4)
feature_list = [ 'trip_distance', 'fare_amount', 'extra', 'temp', 'tolls_amount', 'tip_amount']
plt.figure(figsize=(24,16))
for i in range(1,7):
    plt.subplot(230+i)
    plt.scatter(X[feature_list[i-1]], X.total_amount, c='b', alpha=0.4, label=feature_list[i-1])
    plt.xlabel(feature_list[i-1], fontsize=20)
    plt.ylabel('total_amount', size=20)
    plt.xticks(size=16)
    plt.yticks(fontsize=16)
plt.savefig('scalar.png')
plt.show()

# Remove outliers
box_data = X.loc[:, ['trip_distance', 'fare_amount', 'tolls_amount', 'tip_amount']]
plt.figure(figsize=(10,6))
ax1 = sns.boxplot(x="variable", y="value", data=pd.melt(box_data.astype(np.float)), color='pink')
ax1.xaxis.label.set_size(16)
plt.xticks(size=16)
ax1.yaxis.label.set_size(16)
plt.yticks(size=16)
# plt.savefig('xiangxing.png')
plt.show()

Q1 = box_data.quantile(0.3)
Q3 = box_data.quantile(0.7)
IQR = Q3 - Q1
box_data = box_data[~((box_data < (Q1 - 1.5 * IQR)) |(box_data > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.figure(figsize=(10,6))
ax1 = sns.boxplot(x="variable", y="value", data=pd.melt(box_data.astype(np.float)), color='pink')
ax1.xaxis.label.set_size(16)
plt.xticks(size=16)
ax1.yaxis.label.set_size(16)
plt.yticks(size=16)
plt.savefig('xiangxing.png')
plt.show()

