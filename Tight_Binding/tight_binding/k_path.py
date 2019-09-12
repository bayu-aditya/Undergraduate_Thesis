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

def k_path_custom(k_point_selection, n):
    k = np.array(k_point_selection)
    k_point = [[k[0,0], k[0,1], k[0,2]]]
    for i in range(len(k)-1):
        x = np.linspace(k[i, 0], k[i+1, 0], n[i])
        y = np.linspace(k[i, 1], k[i+1, 1], n[i])
        z = np.linspace(k[i, 2], k[i+1, 2], n[i])
        for idx in range(1,n[i]):
            k_point.append([x[idx], y[idx], z[idx]])
    k_point = np.array(k_point)
    return k_point


def k_mesh_orthorombic(n1, n2, n3, a, b, c):
    """titik k-mesh untuk struktur atom orthorombic di lattice vektor sepanjang a,b,c
    
    Arguments:
        n1 {int} -- banyak titik di sumbu X
        n2 {int} -- banyak titik di sumbu Y
        n3 {int} -- banyak titik di sumbu Z
        a {float} -- panjang sumbu X
        b {float} -- panjang sumbu Y
        c {float} -- panjang sumbu Z
    
    Returns:
        numpy.float64 -- k-mesh untuk struktur kristal orthorombic
    """
    grid = []
    x = np.linspace(-np.pi/a, np.pi/a, n1)
    y = np.linspace(-np.pi/b, np.pi/b, n2)
    z = np.linspace(-np.pi/c, np.pi/c, n3)
    for i in x:
        for j in y: 
            for k in z:
                grid.append([i,j,k])
    return np.array(grid, dtype=np.float64)