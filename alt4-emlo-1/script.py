import pandas as pd
import os

def combine_final_columns(input_files, output_file):
    # List to store the final columns
    final_columns = []
    
    # Read each input file and extract the final column
    for file in input_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Get the last column
            last_column = df.iloc[:, -1]
            
            # Use the filename (without extension) as the column name
            column_name = os.path.splitext(os.path.basename(file))[0]
            
            # Add to our list as a Series with the filename as column name
            final_columns.append(pd.Series(last_column.values, name=column_name))
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Combine all final columns into a single DataFrame
    result = pd.concat(final_columns, axis=1)
    
    # Save to output CSV
    result.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

# Example usage
input_files = [
    'alt4-emlo-1/Anger_epoch_40_pred_eng_a_2025-01-17_18_37_10.csv',
    'alt4-emlo-1/Fear_epoch_40_pred_eng_a_2025-01-17_18_37_09.csv',
    'alt4-emlo-1/Joy_epoch_40_pred_eng_a_2025-01-17_18_37_09.csv',
    'alt4-emlo-1/Sadness_epoch_40_pred_eng_a_2025-01-17_18_37_09.csv',
    'alt4-emlo-1/Surprise_epoch_40_pred_eng_a_2025-01-17_18_37_09.csv'
]

output_file = 'pred_eng.csv'
combine_final_columns(input_files, output_file)