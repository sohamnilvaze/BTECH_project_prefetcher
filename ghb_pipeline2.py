import csv

def parse_trace_file(filename):
    entries = []
    with open(filename, 'r') as f:
        timestamp = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'I':
                pc_str = parts[1][2:]  # Remove "Ox"
                pc = int(pc_str, 16)
                entries.append({'timestamp': timestamp, 'pc': pc, 'addr': None, 'type': 'I', 'tid': None})
            else:
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
    # Get indices for all R/W/R2 entries
    rw_indices = [i for i, e in enumerate(entries) if e['addr'] is not None and e['type'] in ['R', 'W', 'R2']]
    features = []

    for idx, entry_idx in enumerate(rw_indices):
        e = entries[entry_idx]
        feature_row = {
            'Timestamp': e['timestamp'],
            'PC': hex(e['pc']) if e['pc'] is not None else None,
            'Type': e['type'],
            'Address': hex(e['addr']),
        }
        # Deltas with previous n R/W
        for i in range(1, n+1):
            previdx = idx - i
            colname_r = f'Delta_with_{i}_last_read'
            colname_w = f'Delta_with_{i}_last_write'
            delta_r = None
            delta_w = None
            # Look back for ith previous read
            count = 0
            for back in range(previdx, -1, -1):
                eb = entries[rw_indices[back]]
                if eb['type'] in ['R', 'R2']:
                    count += 1
                    if count == 1:
                        delta_r = e['addr'] - eb['addr']
                    if count == i:
                        delta_r = e['addr'] - eb['addr']
                        break
            # Look back for ith previous write
            count = 0
            for back in range(previdx, -1, -1):
                eb = entries[rw_indices[back]]
                if eb['type'] == 'W':
                    count += 1
                    if count == 1:
                        delta_w = e['addr'] - eb['addr']
                    if count == i:
                        delta_w = e['addr'] - eb['addr']
                        break
            feature_row[colname_r] = delta_r
            feature_row[colname_w] = delta_w

        # Deltas with next n R/W
        for i in range(1, n+1):
            nextidx = idx + i
            colname_r = f'Delta_with_{i}_next_read'
            colname_w = f'Delta_with_{i}_next_write'
            delta_r = None
            delta_w = None
            # Look forward for ith next read
            count = 0
            for fwd in range(nextidx, len(rw_indices)):
                ef = entries[rw_indices[fwd]]
                if ef['type'] in ['R', 'R2']:
                    count += 1
                    if count == 1:
                        delta_r = ef['addr'] - e['addr']
                    if count == i:
                        delta_r = ef['addr'] - e['addr']
                        break
            # Look forward for ith next write
            count = 0
            for fwd in range(nextidx, len(rw_indices)):
                ef = entries[rw_indices[fwd]]
                if ef['type'] == 'W':
                    count += 1
                    if count == 1:
                        delta_w = ef['addr'] - e['addr']
                    if count == i:
                        delta_w = ef['addr'] - e['addr']
                        break
            feature_row[colname_r] = delta_r
            feature_row[colname_w] = delta_w

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
    n = int(input("Enter n (number of previous/next read/write deltas): "))
    entries = parse_trace_file(trace_file)
    features = generate_features(entries, n)
    write_features_csv(features, out_csv)
    print(f"Features written to {out_csv}")

if __name__ == "__main__":
    main()
