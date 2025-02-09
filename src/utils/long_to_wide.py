import pandas as pd


input_file = "/local1/bryanzhou008/Dialect/data/text/simplified/ine_v2.csv"
output_file = "/local1/bryanzhou008/Dialect/data/text/simplified_by_word/ine_v2.csv"



df = pd.read_csv(input_file)
df = df.iloc[:, :-2]
grouped_df = df.groupby(['Dialect_Word', 'SAE_Word']).agg({
    'Dialect_Prompt': '; '.join,
    'SAE_Prompt': '; '.join
}).reset_index()
grouped_df.columns = ['Dialect_Word', 'SAE_Word', 'Dialect_Prompts', 'SAE_Prompts']
grouped_df.to_csv(output_file, index=False)


print(f"Processed CSV saved to {output_file}")
