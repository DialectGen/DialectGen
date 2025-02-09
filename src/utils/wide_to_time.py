import csv

def process_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write headers
        headers = ["old", "new"]
        for i in range(1, 7):
            headers.append(f"positive{i}")
            headers.append(f"gt{i}")
        writer.writerow(headers)

        # Process rows
        for row in reader:
            if row[0] == "Dialect_Word":  # Skip header row
                continue

            dialect_word = row[0]
            sae_word = row[1]
            dialect_prompts = row[2].split("; ")
            sae_prompts = row[3].split("; ")

            new_row = [dialect_word, sae_word]
            for d, s in zip(dialect_prompts, sae_prompts):
                new_row.append(d)
                new_row.append(s)

            writer.writerow(new_row)

if __name__ == "__main__":
    input_file = "/local1/bryanzhou008/Dialect/data/text/simplified_by_word/ine_v2.csv"
    output_file = "/local1/bryanzhou008/Dialect/data/text/simplified_time/ine_v2_time.csv"
    process_csv(input_file, output_file)
    print(f"Processed CSV saved to {output_file}")

