import numpy as np
import time
import random
import pickle
from collections import deque

class Layer:
    def __init__(self, name):
        self.name = name
        self.attn_lookup = None
        self.ffn_lookup = None
        self.setup_lookup()
    
    def setup_lookup(self):
        with open('./profiling_logs/attention_exec.pkl', 'rb') as file:
            self.attn_lookup = pickle.load(file)
            print(self.attn_lookup)
        with open('./profiling_logs/ffn_exec.pkl', 'rb') as file:
            self.ffn_lookup = pickle.load(file)
            print(self.ffn_lookup)

    def execute(self,batch_size):
        if batch_size> 3:
            random_skip = random.randint(1,batch_size//2)
            proces_batch_size = batch_size - random_skip
        else:
            proces_batch_size = batch_size
        attn_time = int(self.attn_lookup[proces_batch_size])
        ffn_time = int(self.ffn_lookup[proces_batch_size])
        time.sleep(attn_time+ffn_time)
        return attn_time + ffn_time

class LayerQ:
    def __init__(self, name, skip_probability=0.25):
        self.name = name
        self.skip_probability = skip_probability
        self.attn_lookup = None
        self.ffn_lookup = None
        self.setup_lookup()
        self.queue = deque(maxlen=64)

    def enqueue(self, input_data):
        # Split input data into skipped and processed based on skip_probability
        processed_data = []
        skipped_data = []
        for data in input_data:
            if random.random() > self.skip_probability:
                self.queue.append(data)
                if len(self.queue) == self.queue.maxlen:
                    processed_data.extend(self.execute())
            else:
                skipped_data.append(data)
        
        # Process remaining data in the queue if not full
        if self.queue:
            processed_data.extend(self.execute())
        
        return processed_data, skipped_data
    
    def setup_lookup(self):
        with open('attention_exec.pkl', 'rb') as file:
            self.attn_lookup = pickle.load(file)
            print(self.attn_lookup)
        with open('ffn_exec.pkl', 'rb') as file:
            self.ffn_lookup = pickle.load(file)
            print(self.ffn_lookup)

    def execute(self):
        start_time = time.time()
        processed_data = list(self.queue)
        self.queue.clear()
        time.sleep(self.attn_lookup[63]+self.ffn_lookup[63])
        end_time = time.time()
        return processed_data

class MixtureOfDepthsModel:
    def __init__(self, branches):
        self.branches = branches

    def simulate_execution(self, input_data):
        execution_times = []
        for branch in self.branches:
            start_time = time.time()
            for layer in branch:
                layer.execute(len(input_data))
            end_time = time.time()
            execution_times.append(end_time - start_time)
        return execution_times
    
    def simulate_execution_async(self, input_data):
        execution_times = []
        for branch in self.branches:
            branch_execution_time = 0
            processed_data = input_data
            for layer in branch:
                start_time = time.time()
                processed_data, skipped_data = layer.enqueue(processed_data)
                processed_data.extend(skipped_data)  # Combine processed and skipped data for next layer
                end_time = time.time()
                branch_execution_time += end_time - start_time
            execution_times.append(branch_execution_time)
        return execution_times

    

def generate_requests(num_requests):
    requests = []
    for _ in range(num_requests):
        batch_size = random.randint(1, 64)
        input_data = [np.random.rand(1, 10) for _ in range(batch_size)]
        requests.append(input_data)
    return requests

def process_requests(requests, model, latency_bound):
    processed_requests = 0
    total_execution_time = 0.0

    for request in requests:
        # print("Request")
        start_time = time.time()
        for _ in range(50):  # 150 decoding steps
            for input_data in request:
                execution_times = model.simulate_execution(input_data)
        end_time = time.time()
        total_request_time = end_time - start_time
        print(total_request_time)

        if total_request_time <= latency_bound:
            processed_requests += 1
            # print(processed_requests)
        total_execution_time += total_request_time

    return processed_requests, total_execution_time

def process_requests_with_queue(requests, model, latency_bound):
    processed_requests = 0
    total_execution_time = 0.0

    for req_id,request in enumerate(requests):
        start_time = time.time()
        for _ in range(50):  # 50 decoding steps
            execution_times = model.simulate_execution_async(request)
        end_time = time.time()
        total_request_time = end_time - start_time
        print(total_request_time)

        if total_request_time <= latency_bound:
            processed_requests += 1
        total_execution_time += total_request_time

    return processed_requests, total_execution_time


# Define the model with different branches (paths)
branch1 = [Layer("Layer1")]
branch2 = [Layer("Layer2_1")]
branch3 = [Layer("Layer3_1")]
branch4 = [Layer("Layer4_1")]
branch5 = [Layer("Layer5_1")]
branch6 = [Layer("Layer6_1")]

branch1q = [LayerQ("Layer1")]
branch2q = [LayerQ("Layer2_1")]
branch3q = [LayerQ("Layer3_1")]
branch4q = [LayerQ("Layer4_1")]
branch5q = [LayerQ("Layer5_1")]
branch6q = [LayerQ("Layer6_1")]



# Create the Mixture-of-Depths model
mod_model = MixtureOfDepthsModel([branch1, branch2, branch3,branch4,branch5,branch6])
mod_model_q = MixtureOfDepthsModel([branch1q, branch2q, branch3q,branch4q,branch5q,branch6q])

# Generate requests
num_requests = 50  # Number of requests to simulate
requests = generate_requests(num_requests)

# Define latency bounds to test
latency_bounds = [0.5, 0.7, 2]  # Latency bounds in seconds

# Process requests for each latency bound and calculate throughput
for latency_bound in latency_bounds:
    processed_requests, total_execution_time = process_requests(requests, mod_model, latency_bound)
    # processed_requests, total_execution_time = process_requests_with_queue(requests, mod_model_q, latency_bound)
    throughput = processed_requests / total_execution_time if total_execution_time > 0 else 0
    print(f"Latency bound: {latency_bound:.2f} seconds")
    print(f"Total execution time: {total_execution_time:.2f} seconds")
    print(f"Processed requests: {processed_requests}")
    print(f"Throughput: {throughput:.2f} requests/second")
    print("-------------------------------------------------------------------------------------------")
    # processed_requests, total_execution_time = process_requests(requests, mod_model, latency_bound)
    processed_requests, total_execution_time = process_requests_with_queue(requests, mod_model_q, latency_bound)
    throughput = processed_requests / total_execution_time if total_execution_time > 0 else 0
    print(f"Latency bound: {latency_bound:.2f} seconds")
    print(f"Total execution time: {total_execution_time:.2f} seconds")
    print(f"Processed requests: {processed_requests}")
    print(f"Throughput: {throughput:.2f} requests/second")
    print("")

