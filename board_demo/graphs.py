import matplotlib.pyplot as plt

# Given trial durations
time_intervals = [5, 10, 15]

# Left wall following speeds (distance / time)
left_speeds = {
    5: [43/5, 43.5/5, 43.5/5, 39/5, 45/5],
    10: [99/10, 84/10, 87/10, 85/10, 87.5/10],
    15: [124.5/15, 116/15, 119.5/15, 128/15, 124/15]
}

# Right wall following speeds
right_speeds = {
    5: [41/5, 40.5/5, 41.5/5, 44/5, 41.5/5],
    10: [80.5/10, 75.5/10, 84/10, 78.5/10, 82/10],
    15: [131.5/15, 112.5/15, 118.5/15, 119/15, 119/15]
}

# Flatten data for plotting
x_left, y_left = [], []
x_right, y_right = [], []

for t in time_intervals:
    x_left.extend([t] * len(left_speeds[t]))
    y_left.extend(left_speeds[t])
    x_right.extend([t] * len(right_speeds[t]))
    y_right.extend(right_speeds[t])

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x_left, y_left, color='blue', label='Left Wall Following', alpha=0.7)
plt.scatter(x_right, y_right, color='red', label='Right Wall Following', alpha=0.7)

# Labels and title
plt.xlabel('Trial Duration (seconds)')
plt.ylabel('Speed (cm/sec)')
plt.title('Speed vs. Trial Duration for Left and Right Wall Following')
plt.xticks(time_intervals)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Show plot
plt.show()
