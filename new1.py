import time
import numpy as np
import matplotlib.pyplot as plt

# ----------- Binary Search Implementation -------------
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    steps = 0
    while low <= high:
        steps += 1
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid, steps  # Return index and steps when found
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1, steps  # Return -1 if not found

# ----------- Grover's Search Algorithm Simulation (Approximation for large datasets) -----------
def grover_search_approximation(target_index, size):
    # Approximate number of steps based on Grover's search scaling behavior (sqrt(N))
    grover_steps = int(np.sqrt(size))
    
    # Simulate accuracy decreasing with size
    accuracy = 1 if size <= 64 else 0.8 - (size - 64) * 0.0005
    accuracy = max(accuracy, 0.5)  # Keep accuracy no less than 50%

    # Simulate that Grover's search might not always find the exact target
    if np.random.random() < accuracy:
        return target_index, grover_steps  # Target found
    else:
        return -1, grover_steps  # Target not found

# ----------- Comparison and Benchmarking ----------------
dataset_sizes = list(range(10, 100000, 10000))  # Large dataset sizes for comparison
binary_times = []
grover_times = []
grover_accuracies = []
binary_steps_list = []
grover_steps_list = []

# Test both algorithms for different dataset sizes
for size in dataset_sizes:
    data = np.arange(size)  # Sorted array of integers [0, 1, 2, ..., size-1]
    target = np.random.randint(0, size)  # Random target to search
    
    # Binary Search Benchmark
    start_time = time.time()
    binary_result, binary_steps = binary_search(data, target)
    binary_time = time.time() - start_time
    binary_times.append(binary_time)
    binary_steps_list.append(binary_steps)

    # Grover's Search Approximation (for large dataset)
    start_time = time.time()
    grover_result, grover_steps = grover_search_approximation(target, size)
    grover_time = grover_steps * 1e-6  # Simulating execution time (for demonstration)
    grover_times.append(grover_time)
    grover_steps_list.append(grover_steps)
    grover_accuracies.append(1 if grover_result != -1 else 0)

# ----------- Plotting the Comparison ----------------

# Time comparison (Line Graph)
plt.figure(figsize=(10, 6))
plt.plot(dataset_sizes, binary_times, label="Binary Search Time", color="blue", marker="o")
plt.plot(dataset_sizes, grover_times, label="Grover's Search Time (Approx)", color="orange", marker="o")
plt.xlabel("Dataset Size")
plt.ylabel("Time (seconds)")
plt.title("Execution Time: Binary Search vs Grover's Search (Approximation for Large Datasets)")
plt.legend()
plt.grid(True)
plt.show()

# Accuracy comparison (Bar Chart for large datasets)
labels = [f'Size {size}' for size in dataset_sizes]
x = np.arange(len(dataset_sizes))
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, [1] * len(dataset_sizes), width, label='Binary Search Accuracy', color='blue')
rects2 = ax.bar(x + width/2, grover_accuracies, width, label="Grover's Search Accuracy (Approx)", color='orange')

# Add labels, title, and custom x-axis tick labels
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison: Binary Search vs Grover\'s Search (Approximation for Large Datasets)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

fig.tight_layout()
plt.show()

# ----------- Analysis of Results --------------
for i, size in enumerate(dataset_sizes):
    print(f"Dataset Size: {size}")
    print(f"  - Binary Search: Found at index {binary_result if binary_result != -1 else 'not found'} in {binary_steps_list[i]} steps, Time: {binary_times[i]:.6f} seconds")
    print(f"  - Grover's Search (Approx): Found at index {target if grover_accuracies[i] == 1 else 'not found'} in {grover_steps_list[i]} steps, Time: {grover_times[i]:.6f} seconds")
    print(f"  - Grover's Accuracy: {grover_accuracies[i] * 100:.2f}%")
