import pandas as pd

# Read the CSV file
df = pd.read_csv('pred_eng.csv')

# Replace boolean values with 0 and 1
df = df.replace({False: 0, True: 1})

# Save the modified DataFrame back to CSV
df.to_csv('target.csv', index=False)