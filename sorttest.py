import pandas as pd

# 1. Load the original file
# Make sure master.csv is in the same folder as this script


print(f"Original data loaded: {len(df)} rows.")

# 2. Convert 'Tanggal' to Datetime
# This step is CRUCIAL. It prevents 01/02/2024 from being sorted 
# before 01/01/2024 (which can happen if it's treated as plain text).
df = pd.read_csv('master.csv')
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df_sorted = df.sort_values(by='Tanggal')
# 3. Sort values by Date
# This ensures the timeline is strictly increasing.
df_sorted = df.sort_values(by='Tanggal')

# 4. RESET INDEX (The fix for your 4959 vs 4737 confusion)
# drop=True ensures the old, messy index is deleted.
# Every row gets a new ID starting from 0 in perfect order.
df_sorted = df_sorted.reset_index(drop=True)

# 5. Save to a new CSV file
output_name = 'master_final_sorted.csv'
df_sorted.to_csv(output_name, index=False)

# 6. Verification
print("-" * 30)
print(f"✅ Success! New file saved: {output_name}")
print(f"📅 Start Date: {df_sorted['Tanggal'].min()}")
print(f"📅 End Date:   {df_sorted['Tanggal'].max()}")
print(f"📍 Sample 4959 is now Index 4959.")
print("-" * 30)