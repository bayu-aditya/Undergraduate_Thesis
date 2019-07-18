# author : Bayu Aditya
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from tqdm import tqdm

class density_of_states():
    """
    Rapat keadaan pada kristal Simple Cubic. Menggunakan aproksimasi tight binding.

    Args:
        n (int): grid k-point sebesar (n x n x n)
    """
    def __init__(self, n):
        self.n = n
        self.a = 1          # Angstrom
        k_grid = self._create_grid()
        self.hamiltonian = self._hamiltonian(k_grid)
    

    def get_dos(self, start, stop, n=100):
        """
        Menampilkan grafik rapat keadaan (density of states)

        Args:
            - start (float64): frekuensi awal
            - stop (float64): frekuensi akhir
            - n (int): banyaknya titik frekuensi (default=100)
        """
        frequency = np.linspace(start, stop, n)
        dos = np.zeros_like(frequency)
        for idx in tqdm(range(len(frequency))):
            dos[idx] = self._dos(frequency[idx])
        plt.plot(frequency, dos)
        plt.show()       

    def _create_grid(self):
        """
        Membuat grid k-points 3 dimensi dengan ukuran n x n x n
        
        Returns:
            numpy.ndarray (float64): grid dengan ukuran (n x n x n, 3)
        """
        a, n = self.a, self.n
        grid = []
        x = np.linspace(-np.pi/a, np.pi/a, n)
        y = x
        z = x
        for i in x:
            for j in y: 
                for k in z:
                    grid.append([i,j,k])
        return np.array(grid, dtype=np.float64)


    def _hamiltonian(self, k_points_grid):
        """
        Matriks hamiltonian dengan metode aproksimasi Tight Binding pada kristal Simple Cubic dengan ukuran (2 x 2)

        Args:
            k_points_grid (numpy.ndarray): grid 3 dimensi dengan ukuran (:, 3)

        Returns:
            numpy.ndarray (complex128): Matriks hamiltonian dengan indeks (i,2,2), indeks i adalah k_points_grid ke-i
        """
        param_onsite = 1.0
        param_hopping = 0.1
        a = self.a
        R = [a, a, a]

        hamiltonian = np.empty(
            shape=(len(k_points_grid), 2,2), dtype=np.complex128
            )
        
        kdotr = np.array(0j)
        for idx, k in enumerate(k_points_grid):
            hamiltonian[idx,0,0] = param_onsite
            hamiltonian[idx,1,1] = param_onsite
            # kdotr.imag = np.dot(k, R)
            # hamiltonian[idx,0,1] = -param_hopping*np.exp(kdotr)
            # hamiltonian[idx,1,0] = -param_hopping*np.exp(kdotr)
            hamiltonian[idx,0,1] = -2*param_hopping*(np.cos(k[0]*a) + np.cos(k[1]*a) + np.cos(k[2]*a))
            hamiltonian[idx,1,0] = -2*param_hopping*(np.cos(k[0]*a) + np.cos(k[1]*a) + np.cos(k[2]*a)) 
        return np.array(hamiltonian, dtype=np.complex128)


    def _green_func_matrix(self, frequency, zero_plus=0.01):
        """
        Matriks Green Function fungsi frekuensi

        Args:
            - frequency (float64): frekuensi (eV)
            - zero_plus (float64): opsional, parameter suku imajiner (default=0.01)

        Returns:
            numpy.ndarray (complex128): Matriks Green Function ukuran (:,2,2)
        """
        imag_term = np.array(0j)
        imag_term.imag = zero_plus
        hamiltonian = self.hamiltonian
        green = np.empty_like(hamiltonian)
        
        for idx in range(len(hamiltonian)):
            green[idx,:,:] = inv(
                (frequency+imag_term)*np.eye(2)-hamiltonian[idx,:,:]
                )
        return np.array(green, dtype=np.complex128)


    def _green_func(self, frequency, zero_plus=0.01):
        """
        Matriks Green Function fungsi frekuensi

        Args:
            - frequency (float64): frekuensi (eV)
            - zero_plus (float64): opsional, parameter suku imajiner (default=0.01)

        Returns:
            numpy.complex128: Green Function di frekuensi tertentu
        """
        Green_k = self._green_func_matrix(
            frequency=frequency, zero_plus=zero_plus
            )
        
        sum_green = np.zeros(
            shape=self.hamiltonian.shape[-2:],
            dtype=np.complex128)
        for idx in range(len(Green_k)):
            sum_green += Green_k[idx,:,:]
        return np.trace(sum_green, dtype=np.complex128)


    def _dos(self, frequency):
        """
        Nilai rapat keadaan pada frekuensi tertentu berdasarkan nilai green function.

        Args:
            frequency (float64): Nilai frekuensi
        
        Returns:
            np.float64: Nilai rapat keadaan (Density of States)
        """
        green = self._green_func(frequency=frequency)
        dos = -(1.0/np.pi)*green.imag
        return dos

cls_dos = density_of_states(50)
cls_dos.get_dos(0, 2)
print("Selesai")