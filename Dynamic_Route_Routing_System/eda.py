import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/traffic_data.csv')

# 1. Distribution of Delay
plt.figure(figsize=(8, 6))
sns.histplot(df['Delay'], kde=True, bins=30)
plt.title('Distribution of Delay')
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df[['BusCapacity', 'AvgSpeed', 'DistanceToDestination', 'Delay']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# 3. Boxplot for AvgSpeed to check for outliers
plt.figure(figsize=(8, 6))
sns.boxplot(df['AvgSpeed'])
plt.title('Boxplot for Average Speed')
plt.show()

# 4. Count plot of Traffic Condition
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='TrafficCondition')
plt.title('Count of Traffic Condition')
plt.show()
