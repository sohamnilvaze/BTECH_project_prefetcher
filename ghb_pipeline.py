import csv

def parse_trace_file(filename):
    """
    Parse the Pin trace into a list of dicts with keys:
    'timestamp', 'pc', 'addr', 'type', 'tid'
    """
    entries = []
    with open(filename, 'r') as f:
        timestamp = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if parts[0] == 'I':
                # Instruction line
                # Format: I OxADDR
                pc_str = parts[1][2:]  # remove "Ox"
                pc = int(pc_str, 16)
                # For instructions without access, mark type as None or 'I'
                entries.append({'timestamp': timestamp, 'pc': pc, 'addr': None, 'type': 'I', 'tid': None})
            else:
                # Memory access line: R, W, or R2
                # Format: R tid=0 ip=0xADDR ea=0xADDR
                type_ = parts[0]
                tid = None
                pc = None
                addr = None
                for p in parts[1:]:
                    if p.startswith('tid='):
                        tid = int(p.split('=')[1])
                    elif p.startswith('ip='):
                        pc = int(p.split('=')[1], 16)
                    elif p.startswith('ea='):
                        addr = int(p.split('=')[1], 16)
                entries.append({'timestamp': timestamp, 'pc': pc, 'addr': addr, 'type': type_, 'tid': tid})
            timestamp += 1
    return entries

def generate_features(entries, n):
    """
    Generate features with window size n.
    For each entry except near start/end (due to window), output:
    - Timestamp
    - PC
    - Delta(i-j) for j in n..1
    - Delta(i+j) for j in 1..n
    - Type of current entry
    """
    features = []
    length = len(entries)
    # Filter only memory accesses for delta calculation
    mem_entries = [e for e in entries if e['addr'] is not None]

    # For efficient lookup, create list of addresses only
    addrs = [e['addr'] for e in mem_entries]

    # Map from timestamp in original 'entries' to index in mem_entries
    timestamp_to_memindex = {}
    mem_idx = 0
    for i, e in enumerate(entries):
        if e['addr'] is not None:
            timestamp_to_memindex[i] = mem_idx
            mem_idx += 1
        else:
            timestamp_to_memindex[i] = None

    for i, e in enumerate(entries):
        idx = timestamp_to_memindex.get(i)
        if idx is None:
            # For instructions without address, skip or handle differently
            continue

        if idx < n or idx + n >= len(addrs):
            # Cannot form full window
            continue
        
        feature_row = {'Timestamp': e['timestamp'], 'PC': hex(e['pc']), 'Type': e['type']}
        # Past deltas
        for j in range(n, 0, -1):
            delta = addrs[idx] - addrs[idx - j]
            feature_row[f'Delta(i-{j})'] = delta
            feature_row[f'Address 1_{j}'] = addrs[idx]
            feature_row[f'Address 2_{j}'] = addrs[idx - j]
        # Future deltas
        for j in range(1, n+1):
            delta = addrs[idx + j] - addrs[idx]
            feature_row[f'Delta(i+{j})'] = delta
            feature_row[f'Address 1_{j}'] = addrs[idx + j]
            feature_row[f'Address 2_{j}'] = addrs[idx]
        
        features.append(feature_row)
    return features

def write_features_csv(features, output_file):
    if not features:
        print("No features to write!")
        return
    fieldnames = list(features[0].keys())
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in features:
            writer.writerow(row)

def main():
    trace_file = input("Enter trace file name (e.g., trace.txt): ").strip()
    out_csv = input("Enter output CSV file name (e.g., features.csv): ").strip()
    n = int(input("Enter window size n (number of past and future delta columns): "))
    entries = parse_trace_file(trace_file)
    features = generate_features(entries, n)
    write_features_csv(features, out_csv)
    print(f"Features written to {out_csv}")

if __name__ == "__main__":
    main()
