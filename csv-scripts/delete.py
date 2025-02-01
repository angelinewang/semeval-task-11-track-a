import csv

input_file = '/Users/angwang/semeval-task-11-track-b/example_gold_eng_a.csv'
output_file = '/Users/angwang/semeval-task-11-track-b/example_gold_eng_a-new.csv'

with open(input_file, 'r', newline='', encoding='utf-8') as fin, \
     open(output_file, 'w', newline='', encoding='utf-8') as fout:
    
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    
    for row in reader:
        # If the row has fewer than 2 columns, there's no second column to remove
        if len(row) < 2:
            writer.writerow(row)
        else:
            # Remove the second column (index 1)
            # row[:1] gives everything up to (but not including) index 1, i.e., just the first column
            # row[2:] gives everything from index 2 onward
            new_row = row[:1] + row[2:]
            writer.writerow(new_row)
