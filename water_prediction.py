import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from google.colab import files
import io

# ==========================================
# PART 1: TRAIN THE AI MODEL
# ==========================================
print("🚀 STEP 1: Training the AI Model...")

# Generate dummy training data
dates = pd.date_range(start="2024-01-01", end="2024-12-31")
train_df = pd.DataFrame({'Date': dates})
train_df['Month'] = train_df['Date'].dt.month
train_df['Day_Of_Week'] = train_df['Date'].dt.dayofweek
train_df['Temperature_C'] = 25 + 10 * np.sin((train_df['Month'] / 12) * 3.14) + np.random.normal(0, 2, len(train_df))
train_df['Occupancy_Pct'] = np.where(train_df['Day_Of_Week'] >= 5, 30, 95) 
train_df['Water_Demand_Liters'] = (5000 + (train_df['Temperature_C'] * 100) + (train_df['Occupancy_Pct'] * 50))

features = ['Temperature_C', 'Occupancy_Pct', 'Day_Of_Week', 'Month']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_df[features], train_df['Water_Demand_Liters'])
print("✅ AI Model Trained!")

# ==========================================
# PART 2: UPLOAD YOUR FILE
# ==========================================
print("\n📂 STEP 2: Please Upload your Excel/CSV file now...")
print("(Click the 'Choose Files' button below)")

uploaded = files.upload()

# Get the filename of whatever you just uploaded
filename = next(iter(uploaded))
print(f"   Reading {filename}...")

# Read the file (Handle Excel or CSV automatically)
try:
    if filename.endswith('.csv'):
        input_data = pd.read_csv(io.BytesIO(uploaded[filename]))
    else:
        input_data = pd.read_excel(io.BytesIO(uploaded[filename]))
except Exception as e:
    print(f"❌ Error reading file: {e}")

# ==========================================
# PART 3: FIX COLUMNS & PREDICT
# ==========================================
print("\n🔧 STEP 3: Processing Data...")

# Fix Date
if 'Date' in input_data.columns:
    input_data['Date'] = pd.to_datetime(input_data['Date'])
    input_data['Month'] = input_data['Date'].dt.month
    input_data['Day_Of_Week'] = input_data['Date'].dt.dayofweek

# Fix Occupancy Name
if 'Occupancy_Percent' in input_data.columns:
    input_data.rename(columns={'Occupancy_Percent': 'Occupancy_Pct'}, inplace=True)

# Generate Temperature if missing
if 'Temperature_C' not in input_data.columns:
    print("   -> Simulating missing Temperature data...")
    input_data['Temperature_C'] = 25 + (10 * np.sin((input_data['Month'] / 12) * 3.14))

# Predict
print("🔮 STEP 4: Running Predictions...")
X_new = input_data[features]
input_data['AI_Predicted_Demand'] = model.predict(X_new)

# Add Recommendation
input_data['Pump_Action'] = input_data['AI_Predicted_Demand'].apply(
    lambda x: "⚠️ HIGH DEMAND (Run Aux Pump)" if x > 12000 else "✅ NORMAL (Single Pump)"
)

# ==========================================
# PART 4: DOWNLOAD RESULTS
# ==========================================
output_filename = 'Final_AI_Water_Schedule.xlsx'
input_data.to_excel(output_filename, index=False)

print("\n" + "="*40)
print(f"🎉 SUCCESS! Downloading {output_filename}...")
print("="*40)

files.download(output_filename)
