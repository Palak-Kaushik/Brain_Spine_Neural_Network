import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# First run: Create and save the split
def create_and_save_split(data_path, save_dir="./split_data/"):
    # Read the data
    df = pd.read_csv(data_path)
    
    # Create features and targets
    X = df.drop(["Obesity", "Binary"], axis=1).values
    y_binary = df["Binary"].values
    y_obesity = df["Obesity"].values
    
    # Create the split
    X_train, X_test, y_binary_train, y_binary_test, y_obesity_train, y_obesity_test = train_test_split(
        X, y_binary, y_obesity, test_size=0.2, random_state=42
    )
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Save all splits
    np.save(f"{save_dir}X_train.npy", X_train)
    np.save(f"{save_dir}X_test.npy", X_test)
    np.save(f"{save_dir}y_binary_train.npy", y_binary_train)
    np.save(f"{save_dir}y_binary_test.npy", y_binary_test)
    np.save(f"{save_dir}y_obesity_train.npy", y_obesity_train)
    np.save(f"{save_dir}y_obesity_test.npy", y_obesity_test)
    
    # Save feature names for reference
    feature_names = df.drop(["Obesity", "Binary"], axis=1).columns.tolist()
    pd.Series(feature_names).to_csv(f"{save_dir}feature_names.csv", index=False)
    
    return X_train, X_test, y_binary_train, y_binary_test, y_obesity_train, y_obesity_test

create_and_save_split("./final_data.csv")