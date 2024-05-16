import re
from datetime import datetime
import pickle


def parse_log_file(file_path):
    # Initialize dictionaries to hold batch size and decoding steps
    batch_sizes = {}
    decoding_steps = {}

    # Pattern to capture batch size and decoding steps
    batch_size_pattern = re.compile(r'bsz(\d+)')
    decoding_steps_pattern = re.compile(r'decoding_steps : (\d+)')

    # Read the log file
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Process each line to extract relevant data
    for line in content:
        if "bsz" in line:
            timestamp = line.split(' - ')[0]
            match = batch_size_pattern.search(line)
            if match:
                # Convert timestamp to datetime object
                batch_sizes[datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")] = int(match.group(1))
        elif "decoding_steps" in line:
            timestamp = line.split(' - ')[0]
            match = decoding_steps_pattern.search(line)
            if match:
                # Convert timestamp to datetime object
                decoding_steps[datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")] = int(match.group(1))

    # Match batch sizes and decoding steps based on closest timestamps
    # Create a list of tuples with matched batch sizes and decoding steps
    matched_data = []
    match_dict = {}
    for key, value in decoding_steps.items():
        # Find the closest batch size entry before the decoding step
        closest_batch_time = min(batch_sizes.keys(), key=lambda x: abs(x - key))
        matched_data.append((batch_sizes[closest_batch_time], value))
        match_dict[batch_sizes[closest_batch_time]] = value

    return matched_data,match_dict

# Example usage
file_path = 'decoding_steps.log'
batch_size_decoding_pairs,match_results = parse_log_file(file_path)
print(batch_size_decoding_pairs)
print(match_results)
with open('decoding.pkl', 'wb') as handle:
    pickle.dump(match_results, handle, protocol=pickle.HIGHEST_PROTOCOL)