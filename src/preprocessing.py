from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_data(data):

    # Clean data
    data = data.dropna()
    data.columns = data.columns.str.strip()

    # Detect label column
    label_col = [col for col in data.columns if 'label' in col.lower()][0]

    # Remove infinite values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    # Features & target
    X = data.drop(label_col, axis=1)
    X = X.select_dtypes(include=['number'])   # only numeric

    y = data[label_col]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)