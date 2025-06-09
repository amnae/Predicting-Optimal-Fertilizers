import os 
print(os.getcwd())

from src.data.load_data import load_train_data
from sklearn import linear_model
import numpy as np

reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))
X_train, y_train, X_val, y_val, preprocessor, target_encoder = load_train_data()
reg.fit(X_train, y_train)
print("Best alpha:", reg.alpha_)
reg.score(X_val, y_val)
print("Validation score:", reg.score(X_val, y_val))