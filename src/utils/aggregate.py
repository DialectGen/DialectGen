import csv
from collections import defaultdict

def aggregate_and_append_prompts(input_file, output_file):
    # Use defaultdict to aggregate prompts for each unique Dialect_Word
    prompts_dict = defaultdict(list)

    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            dialect_word = row["Dialect_Word"]
            prompt = row["Prompt"]
            prompts_dict[dialect_word].append(prompt)

    # Write the aggregated prompts to the output CSV
    with open(output_file, 'w', newline='') as outfile:
        # Determine the maximum number of prompts for any Dialect_Word
        max_prompts = max(len(prompts) for prompts in prompts_dict.values())
        
        # Create headers based on the maximum number of prompts
        headers = ["Dialect_Word"]
        for i in range(1, max_prompts + 1):
            headers.extend([f"negative{i}", f"gn{i}"])
        writer = csv.writer(outfile)
        writer.writerow(headers)  # Write header

        for dialect_word, prompts in prompts_dict.items():
            row = [dialect_word]
            for prompt in prompts:
                row.extend([prompt, prompt])  # Append each word two times consecutively
            writer.writerow(row)

if __name__ == "__main__":
    input_file = "/local1/bryanzhou008/Dialect/data/text/polysemy_csvs/AAE_Polysemy.csv"
    output_file = "/local1/bryanzhou008/Dialect/data/text/aggregate_polysemy_csvs/AAE_Polysemy.csv"
    aggregate_and_append_prompts(input_file, output_file)
    print(f"Processed CSV saved to {output_file}")

