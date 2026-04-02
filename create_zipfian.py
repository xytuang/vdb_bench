import numpy as np
import pandas as pd

df = pd.read_parquet('/mydata/spacev1b/test.parquet')
n = len(df)
# Zipfian weights
weights = 1.0 / np.arange(1, n+1)
weights /= weights.sum()
indices = np.random.choice(n, size=n, replace=True, p=weights)
skewed = df.iloc[indices].reset_index(drop=True)
skewed.to_parquet('/mydata/spacev1b/test_zipfian.parquet')
