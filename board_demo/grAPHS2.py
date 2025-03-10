import matplotlib.pyplot as plt
import numpy as np

# Left Wall Following Data (5, 10, and 15 sec trials)
left_5_sec = {
    1: [358, 359, 360, 364, 365, 365, 373, 376, 379, 381, 382, 389, 391, 391, 392, 395, 396],
    2: [360, 362, 361, 361, 360, 361, 363, 362, 363, 365],
    3: [358, 358, 358, 358, 359, 360, 362, 365, 373, 377],
    4: [358, 359, 358, 357, 355, 355, 356, 358, 359],
    5: [354, 358, 358, 358, 357, 357, 358, 359, 361],
}

left_10_sec = {
    6: [348, 353, 356, 358, 358, 361, 363, 365, 365, 368, 373, 370, 366, 363, 361, 359, 358, 357, 356, 352],
    7: [352, 354, 356, 356, 354, 354, 356, 358, 358, 343, 342, 342, 344, 349, 350, 354, 357, 361],
    8: [350, 354, 356, 358, 357, 356, 358, 361, 362, 366, 372, 377, 379, 379, 374, 373, 373],
    9: [354, 360, 358, 359, 359, 359, 361, 362, 363, 364, 365, 365, 366, 367, 365, 367, 368, 365],
    10: [350, 353, 360, 360, 360, 367, 373, 378, 378, 380, 381, 381, 374, 367, 366, 366, 365, 363],
}

left_15_sec = {
    11: [348, 351, 353, 355, 355, 362, 368, 371, 374, 377, 379, 379, 383, 381, 376, 377, 371, 364, 358, 353, 351, 346, 341, 340, 341, 340],
    12: [348, 351, 353, 355, 355, 362, 368, 371, 374, 377, 379, 379, 383, 381, 376, 377, 371, 364, 358, 353, 351, 346, 341, 340, 341, 340],
    13: [352, 354, 355, 356, 354, 355, 355, 356, 357, 358, 360, 344, 344, 346, 346, 347, 350, 353, 354, 356, 357, 359, 360, 359, 360, 361],
    14: [353, 355, 355, 355, 354, 354, 356, 357, 359, 358, 360, 360, 359, 358, 355, 356, 357, 357, 358, 358, 357, 359, 358, 358, 358, 359, 358],
    15: [352, 353, 355, 355, 357, 361, 365, 370, 376, 378, 379, 383, 376, 375, 369, 368, 368, 365, 362, 358, 353, 353, 343, 341, 342],
}

# Right Wall Following Data (5, 10, and 15 sec trials)
right_5_sec = {
    1: [353, 352, 356, 354, 355, 354, 354, 355, 361],
    2: [352, 351, 351, 350, 351, 353, 351, 363, 366, 367],
    3: [355, 355, 353, 352, 351, 353, 354, 356, 360, 357],
    4: [352, 350, 350, 349, 351, 353, 357, 255, 364, 383],
    5: [350, 350, 347, 346, 345, 347, 350, 354, 358, 340],
}

right_10_sec = {
    6: [353, 353, 353, 351, 355, 357, 359, 360, 360, 359, 369, 360, 360, 360, 360, 363, 356],
    7: [356, 356, 351, 352, 352, 347, 348, 354, 349, 349, 349, 344, 350, 352, 352, 358, 361, 363],
    8: [352, 352, 351, 351, 350, 351, 352, 351, 357, 355, 360, 363, 362, 363, 362, 363, 363, 362],
    9: [353, 348, 346, 345, 343, 346, 346, 343, 349, 347, 353, 355, 355, 357, 358, 359, 360, 360],
    10: [357, 356, 352, 349, 346, 350, 346, 349, 349, 349, 351, 348, 352, 355, 357, 361, 363, 364],
}

right_15_sec = {
    11: [354, 354, 348, 346, 344, 344, 343, 342, 348, 347, 355, 358, 359, 360, 361, 362, 362, 362, 361, 361, 381, 381, 383, 385, 386, 386, 379, 375],
    12: [354, 354, 348, 346, 344, 344, 343, 342, 348, 347, 355, 358, 359, 360, 361, 362, 362, 362, 361, 361, 381, 381, 383, 385, 386, 386, 379, 375],
    13: [352, 352, 350, 348, 346, 347, 353, 356, 358, 358, 360, 360, 354, 354, 356, 359, 359, 359, 361, 360, 362, 363, 365, 365, 365, 368],
    14: [353, 354, 352, 357, 361, 356, 356, 355, 357, 358, 358, 359, 359, 359, 361, 362, 362, 360, 359, 359, 361, 364, 365, 367, 372, 375],
    15: [353, 355, 361, 361, 359, 365, 360, 364, 359, 358, 359, 363, 365, 363, 363, 363, 362, 362, 359, 357, 358, 359, 360, 362, 363, 367],
}

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
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Define trial durations (arbitrarily assigned)
trial_durations = {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 
                   6: 10, 7: 10, 8: 10, 9: 10, 10: 10,
                   11: 15, 12: 15, 13: 15, 14: 15, 15: 15}

# Function to normalize oscillations and set x-axis
def normalize_oscillations(trial_data, duration):
    normalized_data = {}
    for trial, values in trial_data.items():
        mean_value = np.mean(values)
        normalized_values = [(v - mean_value) for v in values]  # Normalize oscillations
        num_points = len(values)
        x_values = np.linspace(0, duration, num_points)  # Scale x-axis to the exact duration
        normalized_data[trial] = (x_values, normalized_values)
    return normalized_data

# Normalize each set while keeping x-axis consistent
normalized_trials = {
    "Left 5 sec": normalize_oscillations(left_5_sec, 5),
    "Left 10 sec": normalize_oscillations(left_10_sec, 10),
    "Left 15 sec": normalize_oscillations(left_15_sec, 15),
    "Right 5 sec": normalize_oscillations(right_5_sec, 5),
    "Right 10 sec": normalize_oscillations(right_10_sec, 10),
    "Right 15 sec": normalize_oscillations(right_15_sec, 15),
}

# Function to plot trials
def plot_trials(trial_data, title, ax, duration):
    for trial, (x_values, y_values) in trial_data.items():
        ax.plot(x_values, y_values, marker="o", linestyle="-", label=f"Trial {trial}")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)  # Baseline
    ax.set_xlim(0, duration)  # Ensure x-axis matches time duration
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Oscillation")
    ax.set_title(title)
    ax.legend()

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Plot graphs
plot_trials(normalized_trials["Left 5 sec"], "Left Wall Following - 5 sec", axs[0, 0], 5)
plot_trials(normalized_trials["Left 10 sec"], "Left Wall Following - 10 sec", axs[0, 1], 10)
plot_trials(normalized_trials["Left 15 sec"], "Left Wall Following - 15 sec", axs[0, 2], 15)
plot_trials(normalized_trials["Right 5 sec"], "Right Wall Following - 5 sec", axs[1, 0], 5)
plot_trials(normalized_trials["Right 10 sec"], "Right Wall Following - 10 sec", axs[1, 1], 10)
plot_trials(normalized_trials["Right 15 sec"], "Right Wall Following - 15 sec", axs[1, 2], 15)

plt.tight_layout()
plt.show()