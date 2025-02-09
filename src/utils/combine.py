import csv
from collections import defaultdict

def append_prompts_to_csv(source_csv, target_csv, output_csv):
    # Load the source CSV into a dictionary
    source_dict = {}
    with open(source_csv, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        for row in reader:
            dialect_word = row[0]
            prompts = row[1:]
            source_dict[dialect_word] = prompts

    # Process the target CSV and append prompts
    with open(target_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write header
        header = next(reader)
        max_prompts = max(len(prompts) for prompts in source_dict.values())
        for i in range(1, max_prompts//2 + 1):
            header.extend([f"negative{i}", f"gn{i}"])
        writer.writerow(header)

        # Process rows
        for row in reader:
            dialect_word = row[0]
            if dialect_word in source_dict:
                row.extend(source_dict[dialect_word])
            writer.writerow(row)

if __name__ == "__main__":
    source_csv = "/local1/bryanzhou008/Dialect/data/text/aggregate_polysemy_csvs/ine.csv"  # From the previous script
    target_csv = "/local1/bryanzhou008/Dialect/data/text/expanded_csvs_gpt_basic/ine.csv"
    output_csv = "/local1/bryanzhou008/Dialect/data/text/GPT_ONLY/ine.csv"
    append_prompts_to_csv(source_csv, target_csv, output_csv)
    print(f"Processed CSV saved to {output_csv}")

