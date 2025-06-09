#from sklearn.model_selection import train_test_split
#from sklearn.datasets import load_wine
#from sklearn import linear_model, metrics
#from sklearn.svm import SVC
import pandas as pd
import os
import logging
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Set random state for reproducibility
random_state = 42
# Set validation size
val_size = 0.2

def preprocess_features_extra(df, return_encoders=False):
    processed_df = df.drop(columns=["id"])

    categorical_cols = ["Soil Type", "Crop Type"]
    numeric_cols = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]

    # Target encoding
    onehotencoder = OneHotEncoder(sparse=False)
    processed_df["Fertilizer Encoded"] = onehotencoder.fit_transform(processed_df["Fertilizer Name"])

    # Feature transformation
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    X = preprocessor.fit_transform(processed_df[categorical_cols + numeric_cols])
    y = processed_df["Fertilizer Encoded"].values.toarray()

    return X, y, preprocessor, target_encoder

def preprocess_features(df, return_encoders=False):
    processed_df = df.drop(columns=["id"])

    categorical_cols = ["Soil Type", "Crop Type"]
    numeric_cols = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]

    

    # Feature transformation
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    X = preprocessor.fit_transform(processed_df[categorical_cols + numeric_cols])
    
    # Target encoding
    onehotencoder = OneHotEncoder()
    y = onehotencoder.fit_transform(processed_df["Fertilizer Name"].values.reshape(-1, 1))

    return X, y, preprocessor, onehotencoder

def load_train_data(val = False, base_path = r'data\raw_data'):
    df = pd.read_csv(os.path.join(base_path,'train.csv'))

    X, y, preprocessor, target_encoder = preprocess_features(df, return_encoders=False)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y.toarray(), test_size= val_size, random_state=random_state, stratify=y.toarray()
    )

    return X_train, y_train, X_val, y_val, preprocessor, target_encoder

from torch.utils.data import Dataset
import torch
import numpy as np

class FertilizerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.values).float()
        self.y = torch.from_numpy(y.values).float()

        #self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len
    
if __name__ == "__main__":
    # Load the dataset
    print("Loading Fertilizer dataset...")
    import logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG)
    X_train, y_train, X_val, y_val, preprocessor, target_encoder = load_train_data()
    print("Dataset loaded successfully.")


l = """
wine = load_wine()

X = wine.data[:, :2]
y = wine.target

from sklearn.preprocessing import OneHotEncoder
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=26)
y_train_onehot = OneHotEncoder().fit_transform(y_train.reshape(-1,1)).toarray()
y_test_onehot = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).toarray()

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len
"""