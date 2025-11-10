import random

# Function to generate a uniform random number in [0,1]
def generate_uniform():
    return random.uniform(0, 1)

# Define the function f(x)
def f(x):
    return x**2

# Generate 10 random samples and evaluate f(x)
samples = [generate_uniform() for _ in range(10)]
f_values = [f(x) for x in samples]

# Display samples and function values
for i, (x, fx) in enumerate(zip(samples, f_values), 1):
    print(f"Sample {i}: x = {x:.4f}, f(x) = {fx:.4f}")

# Monte Carlo integration estimate of the integral from 0 to 1
# Integral ≈ average of f(x) over samples * (b - a), here b - a = 1 - 0 = 1
I_estimate = sum(f_values) / len(f_values)

print(f"\nEstimated integral I using Monte Carlo method: {I_estimate:.4f}")
