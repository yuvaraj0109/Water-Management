import pandas as pd
import numpy as np

# 1. Read the file you already uploaded
# (We try reading it as Excel first, if that fails, we read as CSV)
filename = 'Untitled spreadsheet.xlsx' 

print(f"Reading {filename}...")
try:
    input_data = pd.read_excel(filename)
except:
    # If the user uploaded a CSV but named it .xlsx by mistake
    input_data = pd.read_csv(filename)

print("Columns in your file:", input_data.columns.tolist())

# 2. FIX THE MISSING COLUMNS AUTOMATICALLY
print("Fixing missing columns for the AI...")

# Fix 1: Ensure Date is recognized as a date
input_data['Date'] = pd.to_datetime(input_data['Date'])

# Fix 2: Create 'Month' and 'Day_Of_Week' from the Date
input_data['Month'] = input_data['Date'].dt.month
input_data['Day_Of_Week'] = input_data['Date'].dt.dayofweek

# Fix 3: Rename 'Occupancy_Percent' to 'Occupancy_Pct' if needed
if 'Occupancy_Percent' in input_data.columns:
    input_data.rename(columns={'Occupancy_Percent': 'Occupancy_Pct'}, inplace=True)

# Fix 4: Create a dummy 'Temperature_C' column since your file doesn't have it
# (We simulate it: Hotter in months 4, 5, 6)
input_data['Temperature_C'] = 25 + (10 * np.sin((input_data['Month'] / 12) * 3.14))

# 3. RUN THE PREDICTION
print("Running AI Prediction...")

# Select the columns the model needs
features = ['Temperature_C', 'Occupancy_Pct', 'Day_Of_Week', 'Month']
X_new = input_data[features]

# Predict
input_data['AI_Predicted_Demand'] = model.predict(X_new)

# 4. ADD RECOMMENDATIONS
input_data['Pump_Suggestion'] = input_data['AI_Predicted_Demand'].apply(
    lambda x: "⚠️ High Demand: Run Aux Pump" if x > 12000 else "✅ Normal: Single Pump"
)

# 5. SAVE AND DOWNLOAD
output_file = 'AI_Water_Results.xlsx'
input_data.to_excel(output_file, index=False)

print("-" * 30)
print(f"✅ SUCCESS! Results saved to {output_file}")
print("-" * 30)

from google.colab import files
files.download(output_file)
