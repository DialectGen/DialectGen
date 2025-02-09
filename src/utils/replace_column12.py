import csv

def check_column_entries(csv_file1, csv_file2):
    # Read column 1 from both files
    with open(csv_file1, 'r') as file1, open(csv_file2, 'r') as file2:
        reader1 = csv.reader(file1)
        reader2 = csv.reader(file2)

        column1_file1 = [row[0] for row in reader1]
        column1_file2 = [row[0] for row in reader2]

    # Check if the columns are the same length
    if len(column1_file1) != len(column1_file2):
        print("The two CSV files have different numbers of rows.")
        raise

    # Compare entries in column 1 of both files
    for i, (entry1, entry2) in enumerate(zip(column1_file1, column1_file2)):
        if entry1 != entry2:
            print(f"Row {i + 1}: {entry1} in {csv_file1} is not the same as {entry2} in {csv_file2}")
            raise

    print(f"Every entry in column 1 of {csv_file1} is exactly the same as the corresponding entry in {csv_file2}.")



def replace_columns(csv_file1, csv_file2):
    # Read columns 1 and 2 from the second file
    with open(csv_file2, 'r') as file2:
        reader2 = csv.reader(file2)
        columns_file2 = [(row[0], row[1]) for row in reader2]

    # Replace columns 1 and 2 in the first file with the columns from the second file
    with open(csv_file1, 'r') as file1:
        reader1 = csv.reader(file1)
        rows_file1 = [row for row in reader1]

    with open(csv_file1, 'w', newline='') as file1:
        writer1 = csv.writer(file1)
        for (col1, col2), row in zip(columns_file2, rows_file1):
            row[0] = col1
            row[1] = col2
            writer1.writerow(row)

if __name__ == "__main__":
    csv_file1 = "/local1/bryanzhou008/Dialect/data/text/GT_ONLY/bre.csv"
    csv_file2 = "/local1/bryanzhou008/Dialect/data/text/by_word_csvs/bre.csv"
    check_column_entries(csv_file1, csv_file2)
    replace_columns(csv_file1, csv_file2)
    print(f"Replaced columns 1 and 2 of {csv_file1} with columns 1 and 2 of {csv_file2}")

