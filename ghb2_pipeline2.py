import pandas as pd

def count_segment_transitions(addresses):
    # Map addresses to code segments based on prefix
    def map_segment(addr):
        if addr.startswith('0x7'):
            return 'stack'
        elif addr.startswith('0x5'):
            return 'code'
        else:
            return 'other'

    segments = [map_segment(addr) for addr in addresses]
    transitions = sum(1 for i in range(1, len(segments)) if segments[i] != segments[i-1])
    return transitions

def process_higher_ghb(input_csv, output_csv, n):
    df = pd.read_csv(input_csv)
    delta_columns = [
        'Delta_with_1_last_read','Delta_with_1_last_write','Delta_with_2_last_read','Delta_with_2_last_write',
        'Delta_with_3_last_read','Delta_with_3_last_write','Delta_with_1_next_read','Delta_with_1_next_write',
        'Delta_with_2_next_read','Delta_with_2_next_write','Delta_with_3_next_read','Delta_with_3_next_write'
    ]

    result_rows = []
    total_rows = len(df)

    for start in range(0, total_rows, n):
        window = df.iloc[start:start+n]
        if len(window) < n:
            break  # skip incomplete window

        output_row = {}

        # For each delta column, create a pattern string: comma-separated deltas as strings
        for col in delta_columns:
            # Convert to string, replace nan/None with 'Null' or similar
            pattern = ",".join(str(x) if pd.notna(x) else 'Null' for x in window[col])
            output_row[f'{col}_pattern'] = pattern

        # Count code segment transitions in 'Address'
        addresses = window['Address'].astype(str).tolist()
        segment_changes = count_segment_transitions(addresses)
        output_row['code_segment_transitions'] = segment_changes

        # Count reads and writes in window Types
        types = window['Type'].astype(str).tolist()
        n_reads = sum(t in ['R', 'R2'] for t in types)
        n_writes = sum(t == 'W' for t in types)
        output_row['n_reads'] = n_reads
        output_row['n_writes'] = n_writes

        result_rows.append(output_row)

    # Create DataFrame and write to CSV
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(output_csv, index=False)
    print(f"Higher order GHB features written to {output_csv}")

def main():
    input_csv = input("Enter input lower GHB CSV filename: ").strip()
    output_csv = input("Enter output higher GHB CSV filename: ").strip()
    n = int(input("Enter window size n: "))
    process_higher_ghb(input_csv, output_csv, n)

if __name__ == '__main__':
    main()
