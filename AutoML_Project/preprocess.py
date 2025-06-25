import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def detect_target_column(df):
    for col in df.columns:
        if col.lower() in ['target', 'label', 'output', 'class']:
            return col
    return df.columns[-1]

from sklearn.model_selection import train_test_split
from preprocess import detect_target_column  # if used

def preprocess_data(csv_path, target_col=None):
    df = pd.read_csv(csv_path)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='any', inplace=True)

    # Automatically detect target column
    if target_col is None:
        target_col = detect_target_column(df)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # One-hot encode all categorical features
    X = pd.get_dummies(X)
    X = X.fillna(0)

    # Encode target ONLY IF it's classification (non-numeric)
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    return train_test_split(X, y, test_size=0.2, random_state=42)
