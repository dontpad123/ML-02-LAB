 # Install the minisom library
# pip install minisom

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt

# Step 1: Load dataset
# Example: Load the Iris dataset
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target  # Not used in unsupervised learning, just for visualization

# Step 2: Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Initialize and train SOM
# Parameters: 7x7 SOM, input_len = number of features
som = MiniSom(x=7, y=7, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(data=X_scaled, num_iteration=100)

# Step 4: Plotting the SOM

plt.figure(figsize=(10, 8))
frequencies = np.zeros((7, 7))

for x in X_scaled:
    w = som.winner(x)
    frequencies[w[0], w[1]] += 1

# Plot the frequency map (how many samples are mapped to each neuron)
plt.pcolor(frequencies.T, cmap='Blues')
plt.colorbar(label='Number of Mappings')
plt.title("Self-Organizing Map - Mapping Frequencies")
plt.show()

# Optional: Visualize class mappings (if labels are known, for illustration only)
plt.figure(figsize=(10, 8))
for i, x in enumerate(X_scaled):
    w = som.winner(x)
    plt.text(w[0], w[1], str(y[i]),
             ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))

plt.title("SOM with Iris Dataset Labels (for visualization)")
plt.xlim(-0.5, 6.5)
plt.ylim(-0.5, 6.5)
plt.grid()
plt.show()
