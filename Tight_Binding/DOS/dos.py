# author : Bayu Aditya
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from tqdm import tqdm
# import tensorflow as tf
import sys

import time

from Tight_Binding.tight_binding.k_path import k_mesh_orthorombic
from Tight_Binding.tight_binding.hamiltonian import multiple_hamiltonian
from Tight_Binding.tight_binding.hamiltonian import hamiltonian_v3
from Tight_Binding.tools.generator import matrix_generator


class dos_orthorombic:
    def dos_generate_hamiltonian(self, n1, n2, n3, filename_output):
        a = self.a
        b = self.b
        c = self.c
        k_mesh = k_mesh_orthorombic(n1, n2, n3, a, b, c)
        ham_dos = multiple_hamiltonian(k_mesh, self.input_hamiltonian, hamiltonian_v3)
        np.save(filename_output, ham_dos)
        print("Hamiltonian Matrix has been save in : ", filename_output)

    # def plot_dos(start, end, num, filename_input):
    def plot_dos(self, start, end, num, filename_input):
        begin = time.time()
        self._load_hamiltonian(filename_input)
        print(time.time() - begin, "Sec. Load hamiltonian")
        
        frequency = np.linspace(start, end, num)
        dos = np.zeros_like(frequency)
        for i, freq in enumerate(frequency):
            print("Process : ", i, "of", len(frequency))
            dos[i] = -(1.0/np.pi)*np.trace(self._sum_green_over_k(freq)).imag
        plt.plot(frequency, dos)
        plt.savefig("dos")


    def plot_dos_gpu(self, start, end, num, filename_input, step=100):
        begin = time.time()
        self._load_hamiltonian(filename_input)
        print(time.time() - begin, "Sec. Load hamiltonian")
        self._initialization_gpu_tensorflow2()

        frequency = tf.linspace(start, end, num)
        dos = np.zeros(shape=frequency.shape)
        for i, freq in enumerate(frequency):
            print("Process : ", i, "of", len(frequency))
            dos[i] = -(1.0/np.pi)*tf.math.imag(tf.linalg.trace(self._sum_green_over_k_gpu(freq, buffer=step)))
        np.savez("DOS_array.npz", frequency.numpy(), dos)
        # plt.savefig("dos_gpu")
        # plt.plot(frequency, dos)


    def _load_hamiltonian(self, filename_input):
        self.hamiltonian = np.load(filename_input)

    def _initialization_gpu_tensorflow2(self):
        gpus = tf.config.experimental.get_visible_devices()
        tf.config.experimental.set_memory_growth(gpus[1], True)


    def _sum_green_over_k_gpu(self, frequency, zero_plus=0.01, buffer=100):
        freq = tf.cast(frequency, tf.complex128)
        green = tf.zeros(shape=self.hamiltonian.shape[-2:], dtype=tf.complex128)
        for ham in matrix_generator(self.hamiltonian, buffer):
            ham_buff = tf.convert_to_tensor(ham).gpu()
            
            identity_buff = tf.eye(ham_buff.shape[-1:][0], dtype=tf.complex128)
            identity_buff = tf.stack([identity_buff]*len(ham_buff)).gpu()

            green_buff = tf.linalg.inv(
                (freq+zero_plus*1j)*identity_buff - ham_buff
            )
            green_buff = tf.math.reduce_sum(green_buff, axis=0)
            green += green_buff
        return green


    def _sum_green_over_k(self, frequency, zero_plus=0.01):
        identity = np.eye(self.hamiltonian.shape[-1:][0])        # matrix 2D
        identity = np.stack([identity]*len(self.hamiltonian))    # matrix 3D
        
        green = np.linalg.inv(
            (frequency+zero_plus*1j)*identity - self.hamiltonian
            )
        green = np.sum(green, axis=0)
        return green


    def dos_generate_greenfunc_tf1(self, start, end, num, filename_input, step=100, zero_plus=0.01):
        begin = time.time()
        self._load_hamiltonian(filename_input)
        print(time.time() - begin, "Sec. Load hamiltonian")
        N = self.hamiltonian.shape[-1:][0]

        frequency = np.linspace(start, end, num)
        dos = np.zeros_like(frequency)
        pdos = np.zeros(shape=(len(frequency), N))

        hamiltonian = tf.placeholder(tf.complex128, shape=[None, N, N])
        freq = tf.placeholder(tf.complex128)
        with tf.device('/gpu:0'):
            identity = tf.eye(N, dtype=tf.complex128)
            green_buff = tf.linalg.inv(
                (freq+zero_plus*1j)*identity - hamiltonian
            )
            green_buff = tf.imag(tf.diag_part(tf.reduce_sum(green_buff, axis=0)))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            for i, omega in enumerate(frequency):
                begin = time.time()
                green_diag = np.zeros(shape=(N))
                for ham in matrix_generator(self.hamiltonian, step):
                    green_batch = sess.run(green_buff, {hamiltonian: ham, freq: omega})
                    green_diag += green_batch
                pdos[i] = -(1.0/np.pi)*green_diag 
                dos[i] = -(1.0/np.pi)*green_diag.sum()
                print(time.time() - begin, "Sec. Frequency", i+1, "of", len(frequency))
        
        np.savez("DOS_array.npz", frequency, dos, pdos)


    def reduce_dos_band(self, low, high, limit, vline, filename_dos, filename_band):
        self._initialization_gpu_tensorflow2()

        data = np.load(filename_dos)
        ham_band = np.load(filename_band)

        frequency = data['arr_0']
        total_dos = data['arr_1']
        partial_dos = data['arr_2']

        frequency_range = frequency[
            np.where(np.logical_and(frequency>=low, frequency<=high))
            ]
        total_dos_range = total_dos[
            np.where(np.logical_and(frequency>=low, frequency<=high))
            ]
        partial_dos_range = partial_dos[
            np.where(np.logical_and(frequency>=low, frequency<=high))
        ]
        max_total = np.max(total_dos_range)
        max_partial = np.max(partial_dos_range)

        reduce_idx = []

        plt.subplot(2,3,1)
        plt.plot(frequency_range, total_dos_range)
        plt.plot((low, high), (limit*max_total, limit*max_total), 'k--')
        plt.xlim([low, high])
        plt.ylim([0,max_total])

        plt.subplot(2,3,2)
        for i in range(partial_dos_range.shape[-1:][0]):
            plt.plot(frequency_range, partial_dos_range[:,i])
            if (np.max(partial_dos_range[:, i]) < limit*max_partial):
                reduce_idx.append(i)
        plt.plot((low, high), (limit*max_partial, limit*max_partial), 'k--')
        plt.xlim([low, high])
        plt.ylim([0, max_partial])

        print("================================")
        print("length : ", len(reduce_idx), ", limit :", limit)
        print("{:4} {:1} {:5} {:8} {:8} {:8}".format("Atom", "n", "orbtl", "posX", "posY", "posZ"))
        for i in reduce_idx:
            print("{:4} {:1d} {:5} {:8.4f} {:8.4f} {:8.4f}" .format(self.orbital.atom[i], self.orbital.n[i], self.orbital.orbital[i], self.orbital.posX[i],  self.orbital.posY[i], self.orbital.posZ[i]))
        print("================================")

        plt.subplot(2,3,3)
        for i in reduce_idx:
            plt.plot(frequency_range, partial_dos_range[:,i])
        plt.plot((low, high), (limit*max_partial, limit*max_partial), 'k--')
        plt.xlim([low, high])
        plt.ylim([0, max_partial])
        plt.show()


        ham_band = np.delete(ham_band, reduce_idx, axis=1)
        ham_band = np.delete(ham_band, reduce_idx, axis=2)

        ham_d = tf.convert_to_tensor(ham_band)
        eig_d = tf.linalg.eigvalsh(ham_d)
        eig = eig_d.numpy()

        # plt.subplot(2,1,2)
        fermi = 10.83
        for i in range(eig.shape[-1:][0]):
            plt.plot(eig[:,i]-fermi, 'k-')
        for i in vline:
            plt.axvline(i, color='black')
        plt.hlines(0.0, 0, len(eig)-1, linestyles="dashed")
        plt.xlim([0, len(eig)-1])
        plt.ylim([-1.0, 2.0])
        plt.ylabel('$E$ - $E_f$')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.show()
        
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

# cls_dos = density_of_states(50)
# cls_dos.get_dos(0, 2)
# print("Selesai")