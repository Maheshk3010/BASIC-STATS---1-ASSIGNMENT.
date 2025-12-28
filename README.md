# Data Science Assignment

This repository contains my Data Science assignment on basic statistical analysis.

## Python Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/sample_data/sales_data_with_discounts.csv')
df.head()

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

print("Numerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)

mean = df[num_cols].mean()
median = df[num_cols].median()
mode = df[num_cols].mode().iloc[0]
std = df[num_cols].std()

stats_df = pd.DataFrame({
    "Mean": mean,
    "Median": median,
    "Mode": mode,
    "Standard Deviation": std
})

stats_df

df[num_cols].hist(figsize=(12,10))
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=90)
plt.show()

for col in cat_cols:
    plt.figure(figsize=(6,4))
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Bar Chart of {col}")
    plt.show()

df_standardized = df.copy()

for col in num_cols:
    df_standardized[col] = (df[col] - df[col].mean()) / df[col].std()

df_standardized[num_cols].head()

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df_encoded.head()

print("Original shape:", df.shape)
print("After Encoding:", df_encoded.shape)
