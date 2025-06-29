import pandas as pd
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Convert to a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save as CSV
df.to_csv('irisdataset.csv', index=False)

print("✅ 'irisdataset.csv' has been saved.")
