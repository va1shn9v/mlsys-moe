import re
from collections import defaultdict
import pickle

def parse_log_file(file_path):
    # Regular expressions to capture necessary data
    feedforward_time_pattern = re.compile(r'FeedForward execution time for (\d+): ([\d.]+) seconds')
    attention_time_pattern = re.compile(r'Attention execution time for (\d+): ([\d.]+) seconds')

    # Dictionaries to hold total times and counts for averaging
    feedforward_times = defaultdict(lambda: {'total_time': 0, 'count': 0})
    attention_times = defaultdict(lambda: {'total_time': 0, 'count': 0})

    with open(file_path, 'r') as file:
        for line in file:
            feedforward_match = feedforward_time_pattern.search(line)
            attention_match = attention_time_pattern.search(line)

            if feedforward_match:
                batch_size = int(feedforward_match.group(1))
                time = float(feedforward_match.group(2))
                feedforward_times[batch_size]['total_time'] += time
                feedforward_times[batch_size]['count'] += 1

            if attention_match:
                batch_size = int(attention_match.group(1))
                time = float(attention_match.group(2))
                attention_times[batch_size]['total_time'] += time
                attention_times[batch_size]['count'] += 1

    # Calculating average times
    average_feedforward_times = {k: v['total_time'] / v['count'] for k, v in feedforward_times.items()}
    average_attention_times = {k: v['total_time'] / v['count'] for k, v in attention_times.items()}

    return average_feedforward_times, average_attention_times

# Example usage
file_path = 'execution_logger_final.log'
average_feedforward, average_attention = parse_log_file(file_path)
print("Average Feedforward Times:", average_feedforward)
print("Average Attention Times:", average_attention)
with open('attention_exec.pkl', 'wb') as handle:
    pickle.dump(average_attention, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('ffn_exec.pkl', 'wb') as handle:
    pickle.dump(average_feedforward, handle, protocol=pickle.HIGHEST_PROTOCOL)