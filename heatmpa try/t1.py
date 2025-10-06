import numpy as np
import matplotlib.pyplot as plt

# Load the NDVI map from the .npy file
ndvi_map = np.load('heatmpa try/Agra_1_2018_ndvi_heatmap.npy')  # Update the path if needed

# Check the shape and data type of the loaded NDVI map
print(f"NDVI Map Shape: {ndvi_map.shape}")
print(f"NDVI Map Dtype: {ndvi_map.dtype}")

# Plot the raw NDVI values as they are (original data)
plt.figure(figsize=(10, 6))
plt.imshow(ndvi_map, interpolation='none', cmap='RdYlGn')  # No interpolation, 'viridis' just for clarity
# plt.imshow(ndvi_map, interpolation='none', cmap='YlGn')  # No interpolation, 'viridis' just for clarity
plt.title('Raw NDVI Map (Original Values)')
plt.colorbar()  # Add a colorbar to show the range of NDVI values
plt.show()
