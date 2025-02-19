import csv

def calculate_stats(file_path):
    text_lengths = []
    
    # Read the CSV file
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        
        # Calculate length of each text and store in list
        for row in csv_reader:
            text_length = len(row['text'])
            text_lengths.append(text_length)

    # Calculate statistics
    average_length = sum(text_lengths) / len(text_lengths)
    min_length = min(text_lengths)
    max_length = max(text_lengths)
    length_range = max_length - min_length
    
    return average_length, min_length, max_length, length_range

# File paths
files = [
    'public_data_test/track_a/train/eng.csv',
    'public_data_test/track_a/dev/eng.csv',
    'public_data_test/track_a/test/eng.csv'
]

# Process each file
for file_path in files:
    print(f"\nStatistics for {file_path}:")
    avg, min_len, max_len, range_len = calculate_stats(file_path)
    print(f"Average text length: {avg:.2f} characters")
    print(f"Minimum length: {min_len} characters")
    print(f"Maximum length: {max_len} characters")
    print(f"Range: {range_len} characters")