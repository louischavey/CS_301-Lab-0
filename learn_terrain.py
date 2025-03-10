import math
import numpy as np
import pandas as pd
import ast
from matplotlib import pyplot as plt  # Corrected import for pyplot
import seaborn as sns
from collections import Counter
import random

# from ros_robot_controller_sdk import *
# other options for sensor data:
    # Accelerometers for angular acceleration change
    # Sense vibration somehow


training_data = []

def euclidean_distance(v1, v2):  # btwn values, not whole arrays
    return math.sqrt(abs(v1 - v2) ** 2)

def knn_classify(test_velocity, k=3):
    distances = [(euclidean_distance(test_velocity, velocity), surface) for velocity, surface in training_data]
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    surface_counts = {}
    for _, surface in k_nearest:
        surface_counts[surface] = surface_counts.get(surface, 0) + 1
    return max(surface_counts, key=surface_counts.get)

# Step 0: Cleaning and preprocessing
csv_filename = "Lab3TrialData.csv"
df = pd.read_csv(csv_filename)

# Function to compute standard deviation for accelerometer and gyro data
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

# Convert IMU and Velocity columns from string representation to list of lists
df["IMU Readings"] = df["IMU Readings"].apply(ast.literal_eval)
df["Velocity Readings"] = df["Velocity Readings"].apply(ast.literal_eval)

# Compute standard deviations for each trial
df[["sd of acc_x", "sd of acc_y", "sd of acc_z", "sd of rotation rate around x", "sd of rotation rate around y", "sd of rotation rate around z"]] = df["IMU Readings"].apply(lambda imu_list: pd.Series(compute_sd(imu_list)))
df["sd_velocity"] = df["Velocity Readings"].apply(compute_velocity_sd)
df["total_velocity"] = df.apply(lambda row: compute_total_velocity(row["Velocity Readings"], row["Time to End"]), axis=1)

# Drop original IMU column if no longer needed
df = df.drop(columns=["IMU Readings", "Velocity Readings"])

# Step 1: Perform EDA to see relationships btwn variables
# terrain_types = df["Terrain Type"].unique()  # Identify unique terrain types

# # Boxplots for each feature grouped by terrain type
# features = ["sd of acc_x", "sd of acc_y", "sd of acc_z", "sd of rotation rate around x", "sd of rotation rate around y", "sd of rotation rate around z", "sd_velocity", "total_velocity"]

# # Creating figure for plotting
# plt.figure(figsize=(14, 10))
# for i, feature in enumerate(features):
#     plt.subplot(3, 3, i + 1)  # Creating a grid of subplots
#     sns.boxplot(data=df, x="Terrain Type", y=feature)
#     plt.title(f"Boxplot of {feature} by Terrain Type")
# plt.tight_layout()
# plt.show()

# # Pairplot to visualize pairwise relationships between features
# sns.pairplot(df, hue="Terrain Type", vars=features, markers=["o", "s", "D"])
# plt.show()

# Step 2: Split df into training (80%) and testing (20%) while preserving label distribution
def stratified_split(df, label_column, test_size=0.2):
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
    
    return train_df, test_df

train_df, test_df = stratified_split(df, "Terrain Type", test_size=0.2)

# Step 3: Implement KNN from scratch
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))

def manhattan_distance(v1, v2):
    return np.sum(np.abs(np.array(v1) - np.array(v2)))

def knn_predict(X_train, y_train, X_test, k=6):
    predictions = []
    for test_point in X_test:
        distances = [(manhattan_distance(test_point, train_point), label) for train_point, label in zip(X_train, y_train)]
        distances.sort(key=lambda x: x[0])
        k_nearest = [label for _, label in distances[:k]]
        predictions.append(Counter(k_nearest).most_common(1)[0][0])
    return predictions

# Prepare data for KNN
X_train = train_df.drop(columns=["Terrain Type", "Trial #"]).values
y_train = train_df["Terrain Type"].values
X_test = test_df.drop(columns=["Terrain Type", "Trial #"]).values
y_test = test_df["Terrain Type"].values

# Step 4a: Graphing accuracy against different k values
# k_values = range(1, 21)
# train_accuracies = []
# test_accuracies = []

# for k in k_values:
#     y_train_pred = knn_predict(X_train, y_train, X_train, k)
#     y_test_pred = knn_predict(X_train, y_train, X_test, k)
    
#     train_accuracy = np.mean(np.array(y_train_pred) == np.array(y_train))
#     test_accuracy = np.mean(np.array(y_test_pred) == np.array(y_test))
    
#     train_accuracies.append(train_accuracy)
#     test_accuracies.append(test_accuracy)
    
#     print(f"k={k}, Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(k_values, train_accuracies, label="Training Accuracy", marker='o')
# plt.plot(k_values, test_accuracies, label="Testing Accuracy", marker='s')
# plt.xlabel("k value")
# plt.ylabel("Accuracy")
# plt.title("KNN Accuracy for Different k Values")
# plt.legend()
# plt.show()

# Step 4b: One-time test performance
y_pred = knn_predict(X_train, y_train, X_test, k=5)
accuracy = np.mean(np.array(y_pred) == np.array(y_test))
print(f"Model Accuracy: {accuracy:.2f}")

# Save processed data
train_df.to_csv("Lab3TrialData_Train.csv", index=False)
test_df.to_csv("Lab3TrialData_Test.csv", index=False)
print("Training and testing datasets saved.")