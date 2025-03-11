import math
import numpy as np
import pandas as pd
import ast
from matplotlib import pyplot as plt  # Corrected import for pyplot
import seaborn as sns
from collections import Counter
import random

# QUESTIONS:
# - Stratified label distribution w/in training, validation, and testing?
# - So if the cross-validation split is just different shuffles, then I need to take the avg of all these different shuffles?
    # - Interpreting results: Pick a k-value just before we see drop off in accuracy b/c it means we're starting to underfit the data in a way that adversely affects accuracy for unseen data?
    # - How much variation is too much variation?

# STEP 0: Preprocessing
# HELPERS
def compute_sd(imu_data_list):
    imu_array = pd.DataFrame(imu_data_list, columns=["ax", "ay", "az", "gx", "gy", "gz"])
    return (
        imu_array[["ax", "ay", "az"]].std().tolist() +
        imu_array[["gx", "gy", "gz"]].std().tolist()
    )

def compute_velocity_sd(velocity_readings):
    velocity_diffs = np.diff(velocity_readings)  # Compute instantaneous velocity
    return np.std(velocity_diffs) if len(velocity_diffs) > 0 else 0

def compute_total_velocity(velocity_readings, time_to_end):
    return velocity_readings[0] / time_to_end if time_to_end > 0 else 0

# csv_filename = "Lab3TrialData.csv"
# df = pd.read_csv(csv_filename)
# df["IMU Readings"] = df["IMU Readings"].apply(ast.literal_eval)
# df["Velocity Readings"] = df["Velocity Readings"].apply(ast.literal_eval)
# df[["sd of acc_x", "sd of acc_y", "sd of acc_z", "sd of rotation rate around x", "sd of rotation rate around y", "sd of rotation rate around z"]] = df["IMU Readings"].apply(lambda imu_list: pd.Series(compute_sd(imu_list)))
# df["sd_velocity"] = df["Velocity Readings"].apply(compute_velocity_sd)
# df["total_velocity"] = df.apply(lambda row: compute_total_velocity(row["Velocity Readings"], row["Time to End"]), axis=1)
# df = df.drop(columns=["IMU Readings", "Velocity Readings"])
# terrain_types = df["Terrain Type"].unique()

# STEP 1
def eda():  # ACTUAL FUNCTION
    # Boxplots for each feature grouped by terrain type
    features = ["sd of acc_x", "sd of acc_y", "sd of acc_z", "sd of rotation rate around x", "sd of rotation rate around y", "sd of rotation rate around z", "sd_velocity", "total_velocity"]
    plt.figure(figsize=(14, 10))
    for i, feature in enumerate(features):
        plt.subplot(3, 3, i + 1)  # Creating a grid of subplots
        sns.boxplot(data=df, x="Terrain Type", y=feature)
        plt.title(f"Boxplot of {feature} by Terrain Type")

    plt.tight_layout()
    plt.show()

# STEP 2: Split df into training (80%) and testing (20%) while preserving label distribution
def stratified_split(df, label_column, test_size=0.2):  # actual function
    train_data = []
    test_data = []
    
    for label in df[label_column].unique():
        subset = df[df[label_column] == label]
        subset_list = subset.values.tolist()
        random.shuffle(subset_list)
        split_idx = int(len(subset_list) * (1 - test_size))
        train_data.extend(subset_list[:split_idx])
        test_data.extend(subset_list[split_idx:])
    
    train_df = pd.DataFrame(train_data, columns=df.columns)
    test_df = pd.DataFrame(test_data, columns=df.columns)

    train_df.to_csv('train_df', index=False)
    test_df.to_csv('test_df', index=False)

    return train_df, test_df

# STEP 2 (v2): Split df into training, validation, and testing
# def stratified_split_with_validation(df, train_size=0.6, val_size=0.2, test_size=0.2):
#     train_data, val_data, test_data = [], [], []
    
#     for label in df["Terrain Type"].unique():
#         subset = df[df["Terrain Type"] == label]
#         subset_list = subset.values.tolist()
#         random.shuffle(subset_list)
#         train_split = int(train_size * len(subset_list))
#         val_split = int((train_size + val_size) * len(subset_list))
        
#         train_data.extend(subset_list[:train_split])
#         val_data.extend(subset_list[train_split:val_split])
#         test_data.extend(subset_list[val_split:])
    
#     train_df = pd.DataFrame(train_data, columns=df.columns)
#     val_df = pd.DataFrame(val_data, columns=df.columns)
#     test_df = pd.DataFrame(test_data, columns=df.columns)
    
#     return train_df, val_df, test_df

# STEP 3: Implement KNN from scratch
# HELPERS
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))

def manhattan_distance(v1, v2):
    return np.sum(np.abs(np.array(v1) - np.array(v2)))

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = [(euclidean_distance(test_point, train_point), label) for train_point, label in zip(X_train, y_train)]  # compute distance btwn test and training point
        distances.sort(key=lambda x: x[0])
        k_nearest = [label for _, label in distances[:k]]
        predictions.append(Counter(k_nearest).most_common(1)[0][0])
    return predictions

# Function to save model
def save_model(train_df, test_df, filename="knn_model"):
    train_df.to_csv(f"{filename}_train.csv", index=False)
    test_df.to_csv(f"{filename}_test.csv", index=False)
    print(f"Model saved as {filename}_train.csv and {filename}_test.csv")

# Function to load model
def load_model(filename="knn_model"):
    train_df = pd.read_csv(f"{filename}_train.csv")
    test_df = pd.read_csv(f"{filename}_test.csv")
    return train_df, test_df

def find_k(train_df, test_df):  # ACTUAL FUNCTION TO RUN
    # prep data for knn
    X_train = train_df.drop(columns=["Terrain Type", "Trial #"]).values
    y_train = train_df["Terrain Type"].values
    X_test = test_df.drop(columns=["Terrain Type", "Trial #"]).values
    y_test = test_df["Terrain Type"].values

    k_values = range(1, 21)
    train_accuracies = []
    test_accuracies = []

    for k in k_values:
        y_train_pred = knn_predict(X_train, y_train, X_train, k)
        y_test_pred = knn_predict(X_train, y_train, X_test, k)
        
        train_accuracy = np.mean(np.array(y_train_pred) == np.array(y_train))
        test_accuracy = np.mean(np.array(y_test_pred) == np.array(y_test))
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"k={k}, Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, train_accuracies, label="Training Accuracy", marker='o')
    plt.plot(k_values, test_accuracies, label="Testing Accuracy", marker='s')
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy for Different k Values")
    plt.legend()
    plt.show()

    # Prompt user to save model
    save_choice = input("Do you want to save this trained model? (yes/no): ").strip().lower()
    if save_choice == "yes":
        save_model(train_df, test_df)
    else:
        print("Model not saved.")

# def run_kfold_cross_validation(df, train_size=0.6, val_size=0.2, test_size=0.2):
#     k_values = range(1, 21)
#     k_accuracies = {k: [] for k in k_values}
    
#     train_df, val_df, test_df = stratified_split_with_validation(df, train_size, val_size, test_size)
    
#     X_train = train_df.drop(columns=["Terrain Type", "Trial #"]).values
#     y_train = train_df["Terrain Type"].values
#     X_val = val_df.drop(columns=["Terrain Type", "Trial #"]).values
#     y_val = val_df["Terrain Type"].values
#     X_test = test_df.drop(columns=["Terrain Type", "Trial #"]).values
#     y_test = test_df["Terrain Type"].values
    
#     train_accuracies = []
#     test_accuracies = []
    
#     for k in k_values:
#         y_val_pred = knn_predict(X_train, y_train, X_val, k)
#         val_accuracy = np.mean(np.array(y_val_pred) == np.array(y_val))
#         train_accuracy = np.mean(knn_predict(X_train, y_train, X_train, k) == np.array(y_train))
#         test_accuracy = np.mean(knn_predict(X_train, y_train, X_test, k) == np.array(y_test))
        
#         k_accuracies[k].append(val_accuracy)
#         train_accuracies.append(train_accuracy)
#         test_accuracies.append(test_accuracy)
    
#     mean_accuracies = {k: np.mean(k_accuracies[k]) for k in k_values}
    
#     plt.figure(figsize=(10, 5))
#     plt.plot(k_values, list(mean_accuracies.values()), marker='o', label='Validation Accuracy')
#     plt.plot(k_values, train_accuracies, marker='s', linestyle='dashed', label='Training Accuracy')
#     plt.plot(k_values, test_accuracies, marker='^', linestyle='dotted', label='Testing Accuracy')
#     plt.xlabel("k value")
#     plt.ylabel("Accuracy")
#     plt.title("KNN Accuracy for Different k Values using Euclidean Distance")
#     plt.legend()
#     plt.show()

def predict_single_observation(observation, train_df, k=5):
    # Extract features from the new observation
    imu_data = ast.literal_eval(observation["IMU Readings"]) 
    velocity_data = ast.literal_eval(observation["Velocity Readings"])
    
    # Compute derived features for new observation
    sd_values = compute_sd(imu_data)
    sd_velocity = compute_velocity_sd(velocity_data)
    total_velocity = compute_total_velocity(velocity_data, observation["Time to End"])
    
    # Format into a feature array (same structure as X_train)
    feature_vector = np.array(sd_values + [sd_velocity, total_velocity]).reshape(1, -1)

    # Prepare training data (drop "Terrain Type" column)
    feature_columns = ["sd of acc_x", "sd of acc_y", "sd of acc_z",
                        "sd of rotation rate around x", "sd of rotation rate around y", "sd of rotation rate around z",
                        "sd_velocity", "total_velocity"]
    
    X_train = train_df[feature_columns].values  # Train features only
    y_train = train_df["Terrain Type"].values   # Train labels

    # Predict using KNN
    prediction = knn_predict(X_train, y_train, feature_vector, k)[0]
    
    return prediction


# terrain_train_df, terrain_test_df = stratified_split(df, "Terrain Type", test_size=0.2)
# find_k(terrain_train_df, terrain_test_df)

def demo_test():
    demo_train_df, demo_test_df = load_model()
    # load demo csv
    demo_csv = pd.read_csv("DemoData.csv")
    print("\nPredictions for each observation in the dataset (k=4):")
    for index, row in demo_csv.iterrows():
        prediction = predict_single_observation(row, demo_train_df, k=4)
        print(f"Trial {row['Trial #']} - Predicted Terrain Type: {prediction}")

demo_test()
    