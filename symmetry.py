import numpy as np
import time
import os

os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

original = np.full((10000, 10000), 9)
# original =  np.random.rand(10000, 10000)

start_time_transpose = time.time()
transposed = np.transpose(original)
end_time_transpose = time.time()

time_taken_transpose = end_time_transpose - start_time_transpose
print(f"\nTime taken for transpose: {time_taken_transpose} seconds")

start_time_division = time.time()
result = transposed / 3
end_time_division = time.time()

time_taken_division = end_time_division - start_time_division
print(f"\nTime taken for division: {time_taken_division} seconds")
print(result)
