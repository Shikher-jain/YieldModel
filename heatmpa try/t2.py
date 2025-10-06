import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load the NDVI map from the .npy file
ndvi_map = np.load('heatmpa try/Agra_1_2018_ndvi_heatmap.npy')  # Update the path if needed

# Check the shape and data type of the loaded NDVI map
print(f"NDVI Map Shape: {ndvi_map.shape}")
print(f"NDVI Map Dtype: {ndvi_map.dtype}")

# Define the thresholds for categorizing the NDVI values
low_threshold = 0.25   # Below this value, we consider it low vegetation
high_threshold = 0.6  # Above this value, we consider it high vegetation

# Create the masks based on thresholds
mask_low = ndvi_map < low_threshold      # Low vegetation (unhealthy/absent vegetation)
mask_medium = (ndvi_map > 0.2) & (ndvi_map < 0.5)  # Moderate vegetation
mask_high = ndvi_map >= high_threshold   # High vegetation (healthy vegetation)

# Plot the NDVI map and the masks
plt.figure(figsize=(15, 10))

# Plot Raw NDVI Map
plt.subplot(2, 3, 1)
plt.imshow(ndvi_map, interpolation='none', cmap='RdYlGn')
plt.title('Raw NDVI Map')
plt.colorbar()

# Plot Low Yield Mask
plt.subplot(2, 3, 2)
plt.imshow(mask_low, interpolation='none', cmap='Reds')
plt.title('Low Yield Mask (Red)')
plt.colorbar()

yellow_cmap = LinearSegmentedColormap.from_list("Yellow", ["#F4F4EF", "#FFD700"])  # light yellow to dark yellow

# Plot Medium Yield Mask
plt.subplot(2, 3, 3)
plt.imshow(mask_medium, interpolation='none', cmap=yellow_cmap)
plt.title('Medium Yield Mask (Yellow)')
plt.colorbar()

# Plot High Yield Mask
plt.subplot(2, 3, 4)
plt.imshow(mask_high, interpolation='none', cmap='Greens')
plt.title('High Yield Mask (Green)')
plt.colorbar()


# combined_mask = mask_low + mask_medium + mask_high  # Just to visualize the combination of masks
combined_mask = np.stack([mask_low, mask_medium, mask_high], axis=-1).astype(np.float64)

# combined_mask = np.stack([mask_low, mask_medium, mask_high], axis=-1).astype(np.float32)

plt.subplot(2, 3, 5)
plt.imshow(combined_mask, interpolation='none', cmap="RdYlBu")
plt.title('Combined Mask')
plt.colorbar()

plt.tight_layout()
plt.show()
