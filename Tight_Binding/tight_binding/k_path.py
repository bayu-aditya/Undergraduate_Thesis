# author : Bayu Aditya
import numpy as np

def k_path(k_point_selection, n):
    k = np.array(k_point_selection)
    k_point = [[k[0,0], k[0,1], k[0,2]]]
    for i in range(len(k)-1):
        x = np.linspace(k[i, 0], k[i+1, 0], n)
        y = np.linspace(k[i, 1], k[i+1, 1], n)
        z = np.linspace(k[i, 2], k[i+1, 2], n)
        for idx in range(1,n):
            k_point.append([x[idx], y[idx], z[idx]])
    k_point = np.array(k_point)
    return k_point

if __name__ == "__main__":
    a = 1.0
    pi = np.pi
    k = np.array(
        [[0.0, 0.0, 0.0],
         [0.0, pi/a, 0.0],
         [pi/a, pi/a, 0.0],
         [0.0, 0.0, 0.0],
         [pi/a, pi/a, pi/a]]
         )
    k_p = k_path(k, 10)
    print(k_p)