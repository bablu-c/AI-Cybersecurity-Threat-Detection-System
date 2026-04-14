import pandas as pd
from src.preprocessing import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model

# Load dataset
data = pd.read_csv("data/cicids.csv")

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train
model = train_model(X_train, y_train)

# Evaluate
evaluate_model(model, X_test, y_test)