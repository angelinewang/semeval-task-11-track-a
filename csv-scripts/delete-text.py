import csv

# Read the input file and write to a new file
with open('gold_data/test_set/eng.csv', 'r', encoding='utf-8') as infile, \
     open('gold_data/test_set/eng_no_text.csv', 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # For each row, write all columns except the second one (index 1)
    for row in reader:
        new_row = [row[0]] + row[2:]  # Combine first column with all columns after the second
        writer.writerow(new_row)

print("Text column removed successfully!")
