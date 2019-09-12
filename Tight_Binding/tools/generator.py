# author Bayu Aditya
import numpy as np

def matrix_generator(matrix, step):
    n = len(matrix)
    start = 0
    while True:
        if (start+step >= n):
            output = matrix[start:n]
            yield output
            break
        else:
            output = matrix[start:start+step]
            yield output
        start += step

if __name__ == "__main__":
    matrix = np.random.uniform(size=(1000,3,3))
    generator = matrix_generator(matrix, 700)
    for i in generator:
        print(i.shape)