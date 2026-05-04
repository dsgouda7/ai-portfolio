import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ml_models import manual_gradient_descent
from ml_evaluation import evaluate_regression

# --------------------------
# 1. Load the dataset
# --------------------------
df = dd.read_csv(
    r".\data\kl.csv",
    assume_missing=True,     # make int columns nullable for dirty data
    blocksize="16MB",        # chunk size; adjust if memory issues
    dtype=str,               # load as string first to avoid parsing errors
    on_bad_lines="skip",     # skip malformed rows
    encoding_errors="ignore" # skip weird encodings
)

# --------------------------
# 2. Clean and convert specific columns
# --------------------------

# Convert height strings like "5'11" → total inches
df['Height'] = df.map_partitions(
    lambda pdf: pdf['Height'].apply(
        lambda x: int(x.split("'")[0]) * 12 + int(x.split("'")[1])
        if isinstance(x, str) else None
    ),
    meta=('Height', 'float32')
)

# Convert weight strings to numeric
# (Assumes format like "187lbs"; might need more robust parsing if inconsistent)
df['Weight'] = df.map_partitions(
    lambda pdf: pdf['Weight'].apply(
        lambda x: pd.to_numeric(x[-3:], errors='coerce') if isinstance(x, str) else None
    ),
    meta=('Weight', 'float64')
)

# Convert all remaining columns to numeric (bad values → NaN)
df = df.map_partitions(lambda pdf: pdf.apply(pd.to_numeric, errors='coerce'))

# Force all numeric columns to float64 to avoid Dask array dtype issues
df = df.map_partitions(lambda pdf: pdf.astype('float64'))

# --------------------------
# 3. Feature selection
# --------------------------
# Keep only numeric columns

# Find columns where all values are NaN across all partitions
all_nan_cols = df.columns[df.isna().all().compute()].tolist()

# Drop those columns
df = df.drop(columns=all_nan_cols)

valid_cols = df.select_dtypes(include=['number']).columns.tolist()

# Drop non-informative ID-like columns
for col in ['ID', 'Unnamed: 0', 'Jersey Number']:
    if col in valid_cols:
        valid_cols.remove(col)

df = df[valid_cols]

# --------------------------
# 5. Train/test split
# --------------------------
y = df['Potential']              # target column
X = df.drop(columns=['Potential'])  # feature columns

y = y.compute()
X = X.compute()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 6. Handle missing values
# --------------------------
# Impute missing values with mean from training data
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)  # only transform test to avoid data leakage

# --------------------------
# 7. Scale features (z-score normalization)
# --------------------------
scaler = StandardScaler()
scaler.fit(X_train)  # fit only on training set

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

theta, y_pred_manual, epochs, losses = manual_gradient_descent(
    X_train_scaled, y_train, X_test_scaled, y_test,
    alpha=0.07, n_epochs=500, print_every=1, plot_loss=True
)

evaluate_regression(y_test, y_pred_manual, plot=True)
