import math
import numpy as np
import pandas as pd
import ast
from matplotlib import pyplot as plt  # Corrected import for pyplot
import seaborn as sns
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

terrain_types = df["Terrain Type"].unique()  # Identify unique terrain types

# Boxplots for each feature grouped by terrain type
features = ["sd of acc_x", "sd of acc_y", "sd of acc_z", "sd of rotation rate around x", "sd of rotation rate around y", "sd of rotation rate around z", "sd_velocity", "total_velocity"]

# Creating figure for plotting
plt.figure(figsize=(14, 10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1)  # Creating a grid of subplots
    sns.boxplot(data=df, x="Terrain Type", y=feature)
    plt.title(f"Boxplot of {feature} by Terrain Type")
plt.tight_layout()
plt.show()

# Pairplot to visualize pairwise relationships between features
sns.pairplot(df, hue="Terrain Type", vars=features, markers=["o", "s", "D"])
plt.show()

# Step 2: