import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            label_encoders[column] = le
    return df, label_encoders
