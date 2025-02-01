import csv

input_file = '../results/track_a/pred_eng_new1.csv'
output_file = '../results/track_a/pred_eng_new2.csv'

with open(input_file, 'r', newline='', encoding='utf-8') as fin, \
     open(output_file, 'w', newline='', encoding='utf-8') as fout:
    
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    
    for row in reader:
        # Swap columns only if row has at least 3 columns
        if len(row) >= 3:
            # Swap second (index 1) and third (index 2) columns
            row[1], row[2] = row[2], row[1]
        # Write the (possibly modified) row
        writer.writerow(row)
