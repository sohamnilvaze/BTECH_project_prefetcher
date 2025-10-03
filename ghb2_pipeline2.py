# import pandas as pd

# def count_segment_transitions(addresses):
#     # Map addresses to code segments based on prefix
#     def map_segment(addr):
#         if addr.startswith('0x7'):
#             return 'stack'
#         elif addr.startswith('0x5'):
#             return 'code'
#         else:
#             return 'other'

#     segments = [map_segment(addr) for addr in addresses]
#     transitions = sum(1 for i in range(1, len(segments)) if segments[i] != segments[i-1])
#     return transitions

# def process_higher_ghb(input_csv, output_csv, n):
#     df = pd.read_csv(input_csv)
#     delta_columns = [
#         'Delta_with_1_last_read','Delta_with_1_last_write','Delta_with_2_last_read','Delta_with_2_last_write',
#         'Delta_with_3_last_read','Delta_with_3_last_write','Delta_with_1_next_read','Delta_with_1_next_write',
#         'Delta_with_2_next_read','Delta_with_2_next_write','Delta_with_3_next_read','Delta_with_3_next_write'
#     ]

#     result_rows = []
#     total_rows = len(df)

#     for start in range(0, total_rows, n):
#         window = df.iloc[start:start+n]
#         if len(window) < n:
#             break  # skip incomplete window

#         output_row = {}

#         # For each delta column, create a pattern string: comma-separated deltas as strings
#         for col in delta_columns:
#             # Convert to string, replace nan/None with 'Null' or similar
#             pattern = ",".join(str(x) if pd.notna(x) else 'Null' for x in window[col])
#             output_row[f'{col}_pattern'] = pattern

#         # Count code segment transitions in 'Address'
#         addresses = window['Address'].astype(str).tolist()
#         segment_changes = count_segment_transitions(addresses)
#         output_row['code_segment_transitions'] = segment_changes

#         # Count reads and writes in window Types
#         types = window['Type'].astype(str).tolist()
#         n_reads = sum(t in ['R', 'R2'] for t in types)
#         n_writes = sum(t == 'W' for t in types)
#         output_row['n_reads'] = n_reads
#         output_row['n_writes'] = n_writes

#         result_rows.append(output_row)

#     # Create DataFrame and write to CSV
#     result_df = pd.DataFrame(result_rows)
#     result_df.to_csv(output_csv, index=False)
#     print(f"Higher order GHB features written to {output_csv}")

# def main():
#     input_csv = input("Enter input lower GHB CSV filename: ").strip()
#     output_csv = input("Enter output higher GHB CSV filename: ").strip()
#     n = int(input("Enter window size n: "))
#     process_higher_ghb(input_csv, output_csv, n)

# if __name__ == '__main__':
#     main()

import pandas as pd
import numpy as np
from collections import Counter
import math
from itertools import groupby

# --- Helper Functions for New Features ---

def count_segment_transitions(addresses):
    """Counts transitions between 'stack', 'code', and 'other' segments."""
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

def calculate_entropy(values):
    """Calculates Shannon Entropy (in bits) for a list of values."""
    if not values:
        return 0.0
    # Calculate frequency of each unique value
    counts = Counter(values)
    total = len(values)
    
    # Calculate probability and entropy
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy

def get_dominant_subsequence(types):
    """Finds the most frequent contiguous subsequence of R/W/R2 types."""
    if not types:
        return 'Null'
    
    subsequences = []
    
    # Generate all possible subsequences (up to length 3, for practicality)
    for length in range(1, min(len(types), 4)):
        for i in range(len(types) - length + 1):
            subsequences.append(",".join(types[i:i+length]))
            
    if not subsequences:
        return 'Null'
        
    # Find the most common one
    counter = Counter(subsequences)
    # Return the subsequence string and its count
    most_common, count = counter.most_common(1)[0]
    return f"{most_common} ({count})"

def calculate_delta_metrics(delta_series):
    """Calculates Most Frequent Delta, Mean Absolute Delta, and Stride Changes."""
    
    # Filter out NaNs (from 'Null' values) and convert to float/int
    valid_deltas = delta_series.dropna().astype(float).astype(int).tolist()
    
    if not valid_deltas:
        return None, None, 0

    # 1. Most Frequent Delta
    counter = Counter(valid_deltas)
    most_frequent_delta = counter.most_common(1)[0][0] if counter else 'Null'

    # 2. Mean Absolute Delta
    mean_absolute_delta = np.mean(np.abs(valid_deltas))

    # 3. Number of Stride Changes (sign change of delta)
    stride_changes = 0
    if len(valid_deltas) > 1:
        # Get the sign of each delta (1 for positive, -1 for negative, 0 for zero)
        signs = np.sign(valid_deltas)
        # Count where the sign changes from the previous one (ignoring 0)
        previous_sign = 0
        for sign in signs:
            if sign != 0 and previous_sign != 0 and sign != previous_sign:
                stride_changes += 1
            if sign != 0:
                previous_sign = sign
                
    return most_frequent_delta, mean_absolute_delta, stride_changes


# --- Main Processing Function ---

def process_higher_ghb(input_csv, output_csv, n):
    df = pd.read_csv(input_csv)
    
    # Ensure Address and PC columns are string-like for set operations and entropy
    df['Address'] = df['Address'].astype(str)
    df['PC'] = df['PC'].astype(str)
    
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

        # --- EXISTING LOGIC & NEW DELTA METRICS (5 metrics per delta column) ---
        for col in delta_columns:
            
            # 1. Delta Pattern (Existing)
            pattern = ",".join(str(x) if pd.notna(x) else 'Null' for x in window[col])
            output_row[f'{col}_pattern'] = pattern
            
            # 2-4. New Delta Metrics
            mfd, mad, nsc = calculate_delta_metrics(window[col])
            
            output_row[f'{col}_most_frequent_delta'] = mfd
            output_row[f'{col}_mean_abs_delta'] = mad
            output_row[f'{col}_n_stride_changes'] = nsc


        # --- EXISTING AGGREGATE METRICS ---

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
        
        # --- NEW AGGREGATE METRICS ---
        
        # 5. Dominant Most Common Subsequence Pattern of Type
        output_row['dominant_type_subsequence'] = get_dominant_subsequence(types)

        # 6. Entropy of Addresses
        # Use the actual address values for entropy
        output_row['entropy_of_addresses'] = calculate_entropy(addresses)

        # 7. Entropy of PC
        # Use the actual PC values for entropy
        pc_list = window['PC'].astype(str).tolist()
        output_row['entropy_of_pc'] = calculate_entropy(pc_list)


        result_rows.append(output_row)

    # Create DataFrame and write to CSV
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(output_csv, index=False)
    print(f"Higher order GHB features written to {output_csv}")

def main():
    input_csv = input("Enter input lower GHB CSV filename: ").strip()
    output_csv = input("Enter output higher GHB CSV filename: ").strip()
    try:
        n = int(input("Enter window size n: "))
        if n < 1:
             raise ValueError("Window size n must be positive.")
    except ValueError as e:
        print(f"Error: Invalid input for n. {e}")
        return

    process_higher_ghb(input_csv, output_csv, n)

if __name__ == '__main__':
    main()