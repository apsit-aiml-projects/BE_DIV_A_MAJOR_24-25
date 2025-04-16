import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
player_df = pd.read_csv("player_coordinates.csv")
shuttle_df = pd.read_csv("shuttle_coordinates.csv")

# Extract X, Y coordinates
player_x, player_y = player_df["X"], player_df["Y"]
shuttle_x, shuttle_y = shuttle_df["X"], shuttle_df["Y"]

# Plot the court boundary (assuming 13.4m x 6.1m full court)
court_width = 610  # Arbitrary scale (6.1m converted for better visibility)
half_court_length = 670  # Half court length (13.4m total, so half is 6.7m)

plt.figure(figsize=(10, 6))
plt.xlim(0, court_width)
plt.ylim(0, half_court_length * 2)  # Full court visualization

# Plot the net at halfway
plt.axhline(y=half_court_length, color="black", linestyle="--", linewidth=2, label="Net")

# Plot player movement
plt.plot(player_x, player_y, marker="o", linestyle="-", color="blue", label="Player Movement")

# Plot shuttlecock movement
plt.plot(shuttle_x, shuttle_y, marker="x", linestyle="--", color="red", label="Shuttlecock Movement")

# Labels and Title
plt.xlabel("Court Width (Arbitrary Units)")
plt.ylabel("Court Length (Arbitrary Units)")
plt.title("Player and Shuttlecock Movement")
plt.legend()

# Show the plot
plt.show()





# import pandas as pd
# import numpy as np

# # Define court dimensions (approximate)
# court_width = 610  # in cm
# court_length = 1340  # in cm
# half_court_length = court_length / 2

# # Number of frames
# num_frames = 50

# # Generate player coordinates (Player stays in one half)
# player_x = np.linspace(100, court_width - 100, num_frames) + np.random.randint(-50, 50, num_frames)
# player_y = np.abs(np.sin(np.linspace(0, 4 * np.pi, num_frames))) * (half_court_length - 100) + 100

# # Generate shuttlecock coordinates (It moves across the entire court)
# shuttle_x = np.random.randint(100, court_width - 100, num_frames)
# shuttle_y = np.linspace(100, court_length - 100, num_frames) + np.random.randint(-50, 50, num_frames)

# # Simulate hits (when player reaches the shuttle in their half, shuttle changes direction)
# for i in range(1, num_frames):
#     if shuttle_y[i] <= half_court_length:  # If shuttle is in player's half
#         dist = np.sqrt((shuttle_x[i] - player_x[i]) ** 2 + (shuttle_y[i] - player_y[i]) ** 2)
#         if dist < 100:  # If player is close enough, simulate a hit
#             shuttle_y[i:] = court_length - shuttle_y[i:]  # Flip trajectory

# # Save to CSV
# player_df = pd.DataFrame({'Frame': range(num_frames), 'X': player_x, 'Y': player_y})
# shuttle_df = pd.DataFrame({'Frame': range(num_frames), 'X': shuttle_x, 'Y': shuttle_y})

# player_df.to_csv("player_coordinates.csv", index=False)
# shuttle_df.to_csv("shuttle_coordinates.csv", index=False)

# # Display first few rows
# player_df.head(), shuttle_df.head()

