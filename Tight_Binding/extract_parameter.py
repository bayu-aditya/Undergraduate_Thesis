# author : Bayu Aditya (2019)
import numpy as np
import pandas as pd

class extract_parameter():
    def __init__(self, hr_files):
        num_init_row = self._check_initial_row(hr_files)
        self.data = pd.read_fwf(
            files_hr, 
            names=["X", "Y", "Z", "A", "B", "Re", "Im"], 
            skiprows=num_init_row
            )

    def get_data(self):
        return self.data
    
    def _check_initial_row(self, hr_files):
        with open(hr_files) as f:
            data = f.readlines()
        for num_row, row in enumerate(data):
            if (len(row.split()) == 7):
                break            
        return num_row