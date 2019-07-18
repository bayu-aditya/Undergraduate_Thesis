# author : Bayu Aditya (2019)
import numpy as np
import pandas as pd

class extract_parameter():
    """
    Ekstrak Parameter dari file output Wannier90 *_hr.dat . Data parameter ditampilkan dalam bentuk DataFrame.

    Args:
        hr_files (str): nama dan lokasi file hr
    """
    def __init__(self, hr_files):
        num_init_row = self._check_initial_row(hr_files)
        self.data = pd.read_fwf(
            hr_files, 
            names=["X", "Y", "Z", "A", "B", "Re", "Im"], 
            skiprows=num_init_row
            )


    def get_data(self):
        """
        Mengambil parameter berbentuk DataFrame

        Returns:
            pandas.DataFrame: Parameter Tight Binding
        """
        return self.data


    def _check_initial_row(self, hr_files):
        """
        Menentukan letak baris awal pembacaan parameter

        Args:
            hr_files (str): nama dan lokasi file hr
        
        Returns:
            int: nomor baris awal pembacaan parameter
        """
        with open(hr_files) as f:
            data = f.readlines()
        for num_row, row in enumerate(data):
            if (len(row.split()) == 7):
                break            
        return num_row