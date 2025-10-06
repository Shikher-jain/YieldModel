import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import requests

# Load NDVI and Sensor Data
files = {
    'ndvi_file': open('data/ndvi/Agra_1_2018_ndvi_heatmap.npy', 'rb'),
    'sensor_file': open('data/sensor/Agra_51_2018_Sensor.npy', 'rb')
}
response = requests.post("http://127.0.0.1:8000/predict", files=files)
yield_pred = response.json()['predicted_yield']
# yield_pred /= 10  # Scaling may not be necessary, you can try without it
print(f"Predicted Yield: {yield_pred}")

# Load the NDVI map from the .npy file
ndvi_map = np.load('heatmpa try/Agra_1_2018_ndvi_heatmap.npy')  # Update the path if needed

# Check the shape and data type of the loaded NDVI map
print(f"NDVI Map Shape: {ndvi_map.shape}")
print(f"NDVI Map Dtype: {ndvi_map.dtype}")

# Define the thresholds for categorizing the NDVI values, dynamically adjusted by yield_pred
low_threshold = 0.25 + 0.25 / yield_pred  # Below this value, we consider it low vegetation
high_threshold = 0.6 - 0.6 / yield_pred  # Above this value, we consider it high vegetation

# Create the masks based on thresholds, influenced by yield_pred
mask_low = ndvi_map < low_threshold  # Low vegetation (unhealthy/absent vegetation)
mask_medium = (ndvi_map > low_threshold + 0.05) & (ndvi_map < high_threshold - 0.05)  # Moderate vegetation
mask_high = ndvi_map >= high_threshold  # High vegetation (healthy vegetation)

# Plot the NDVI map and the masks in a raster form
plt.figure(figsize=(15, 10))

# Plot Raw NDVI Map
plt.subplot(2, 3, 1)
plt.imshow(ndvi_map, cmap='RdYlGn', interpolation='none')
plt.title('Raw NDVI Map (Raster)')
plt.colorbar(label='NDVI Value')

# Plot Low Yield Mask (Low Vegetation)
plt.subplot(2, 3, 2)
plt.imshow(mask_low, cmap='Reds', interpolation='none')
plt.title('Low Yield Mask (Raster) - Red')
plt.colorbar(label='Mask Value (Low)')

# Custom Yellow Colormap for Medium Yield Mask
yellow_cmap = LinearSegmentedColormap.from_list("Yellow", ["#F4F4EF", "#FFD700"])  # light yellow to dark yellow

# Plot Medium Yield Mask (Moderate Vegetation)
plt.subplot(2, 3, 3)
plt.imshow(mask_medium, cmap=yellow_cmap, interpolation='none')
plt.title('Medium Yield Mask (Raster) - Yellow')
plt.colorbar(label='Mask Value (Medium)')

# Plot High Yield Mask (High Vegetation)
plt.subplot(2, 3, 4)
plt.imshow(mask_high, cmap='Greens', interpolation='none')
plt.title('High Yield Mask (Raster) - Green')
plt.colorbar(label='Mask Value (High)')

# Combine the Masks into a Single Mask (Optional visualization of all)
combined_mask = np.zeros_like(ndvi_map, dtype=np.float32)

# Assign unique values for each category
combined_mask[mask_low] = 1  # Low yield
combined_mask[mask_medium] = 2  # Medium yield
combined_mask[mask_high] = 3  # High yield

# Plot Combined Mask
plt.subplot(2, 3, 5)
plt.imshow(combined_mask, cmap=LinearSegmentedColormap.from_list(
    'MaskCmap', ['#FF0000', '#FFD700', '#008000'], N=3), interpolation='none')
plt.title('Combined Mask (Low, Medium, High)')
plt.colorbar(label='Mask Value (Low, Medium, High)')

# Tight layout for better visualization
plt.tight_layout()
plt.show()
