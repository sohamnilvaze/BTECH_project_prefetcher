import pandas as pd
from collections import defaultdict

# def build_markov_chain(csv_file, column_name):
#     """
#     Build a Markov chain transition probability matrix from a column in a CSV file.

#     Args:
#         csv_file (str): Path to the CSV file.
#         column_name (str): Column name containing the sequence (deltas).

#     Returns:
#         dict: Nested dict {state_i: {state_j: probability}} representing the chain.
#     """
#     # Load the column
#     df = pd.read_csv(csv_file)
#     if column_name not in df.columns:
#         raise ValueError(f"Column '{column_name}' not found in {csv_file}")

#     sequence = df[column_name].dropna().tolist()
#     if len(sequence) < 2:
#         raise ValueError("Not enough data to build a Markov chain")

#     # Count transitions
#     transition_counts = defaultdict(lambda: defaultdict(int))
#     for i in range(len(sequence) - 1):
#         current_state = sequence[i]
#         next_state = sequence[i + 1]
#         transition_counts[current_state][next_state] += 1

#     # Normalize to probabilities
#     transition_matrix = {}
#     for state, next_states in transition_counts.items():
#         total = sum(next_states.values())
#         transition_matrix[state] = {ns: count / total for ns, count in next_states.items()}

#     return transition_matrix



def build_markov_chain(csv_files, column_name):
    """
    Build a Markov chain transition probability matrix from a column in a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        column_name (str): Column name containing the sequence (deltas).

    Returns:
        dict: Nested dict {state_i: {state_j: probability}} representing the chain.
    """
    transition_counts = defaultdict(lambda: defaultdict(int))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in {csv_file}")

        sequence = df[column_name].dropna().tolist()
        if len(sequence) < 2:
            print(f"Not enough data to build a Markov chain for {df}")


        # Count transitions
        
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]
            transition_counts[current_state][next_state] += 1

    # Normalize to probabilities
    transition_matrix = {}
    for state, next_states in transition_counts.items():
        total = sum(next_states.values())
        transition_matrix[state] = {ns: count / total for ns, count in next_states.items()}

    return transition_matrix



# Example usage:
if __name__ == "__main__":
    csv_files = ["ghb_1/f1_colmajor.csv","ghb_1/f1_seq.csv","ghb_1/f1_strided.csv"
,"ghb_1/f1_rowmajor.csv","ghb_1/f1_ra.csv","ghb_1/f1_rec.csv","ghb_1/f1_llt.csv"]    
    column_name = input("Enter the name of column:\n")   # e.g. pick any delta column
    output_file = input("Enter the output(.txt) file:\n")
    mc = build_markov_chain(csv_files, column_name)

    print("Transition Matrix:")
    for state, transitions in mc.items():
        print(f"{state} -> {transitions}")
    
    with open(output_file,"w") as f:
        f.write(f"Transition matrix for {column_name}")
        for state , transitions in mc.items():
            f.write(f"{state} -> {transitions}\n")
    
    print(f"Written sucessfully to {output_file}")
