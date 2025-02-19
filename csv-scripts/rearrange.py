import csv

input_file = '/Users/angwang/semeval-task-11-track-a/post-hoc/results-2/track_a/pred_eng.csv'
output_file = '/Users/angwang/semeval-task-11-track-a/post-hoc/results-2/track_a/pred_eng-new.csv'

# Adjust these indices if your CSV has a header row or different column structure
# Original indices:       [0, 1,       2,        3,         4,    5]
# Original column names:  [id, joy, sadness, surprise, fear, anger]
# Desired column order:   [id, anger, fear, joy, sadness, surprise]
# Desired new indices:    [0,    5,     4,   1,     2,        3]

with open(input_file, 'r', newline='', encoding='utf-8') as fin, \
     open(output_file, 'w', newline='', encoding='utf-8') as fout:
    
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    
    for row in reader:
        # Ensure the row has at least 6 columns before reordering:
        if len(row) >= 6:
            # Reorder the row according to the desired new column order
            new_row = [row[0], row[5], row[4], row[1], row[2], row[3]]
            writer.writerow(new_row)
        else:
            # If fewer than 6 columns, write as-is (or handle differently if needed)
            writer.writerow(row)
