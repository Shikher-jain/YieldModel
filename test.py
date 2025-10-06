import requests

files = {
    'ndvi_file': open('data/ndvi/Agra_1_2018_ndvi_heatmap.npy', 'rb'),
    'sensor_file': open('data/sensor/Agra_51_2018_Sensor.npy', 'rb')
}
response = requests.post("http://127.0.0.1:8000/predict", files=files)
print(response.json())
print(response.json()['predicted_yield'])
yield_pred = response.json()['predicted_yield']
yield_pred/= 10  # Scale the yield prediction to match NDVI scale (0-1)
print(f"Predicted Yield: {yield_pred}")
print(f"Predicted Yield: {yield_pred/10}")