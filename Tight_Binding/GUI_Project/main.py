# Author : Bayu Aditya
from PyQt5 import QtWidgets, uic
from PyQt5.Qt import Qt
import numpy as np
import pandas as pd
import logging
import sys
sys.path.append('../')

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from src.variable import jsonio
from src.k_path import k_path_custom

class Ui(QtWidgets.QTabWidget):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(uifile="mainwindow2.ui", baseinstance=self)
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

        # Layout PLOT DOS
        self.lay_plot = self.findChild(QtWidgets.QVBoxLayout, "plotLayoutDos")
        self._canvas_dos = FigureCanvas(Figure(figsize=(5,3)))
        self.lay_plot.addWidget(self._canvas_dos)

        # Input Groupbox
        self.pte_input_dos = self.findChild(QtWidgets.QPlainTextEdit, "plainTextEdit")
        self.pb_browse = self.findChild(QtWidgets.QPushButton, "pushButton_browse")
        self.pb_browse.clicked.connect(self.browsePressed_dos)

        # RANGE GroupBox
        self.le_start = self.findChild(QtWidgets.QLineEdit, "lineEdit_start")
        self.le_stop = self.findChild(QtWidgets.QLineEdit, "lineEdit_stop")
        self.le_intmax = self.findChild(QtWidgets.QLineEdit, "lineEdit_intmax")
        self.le_intmin = self.findChild(QtWidgets.QLineEdit, "lineEdit_intmin")
        self.pb_plot = self.findChild(QtWidgets.QPushButton, "pushButton_plot")
        self.pb_plot.clicked.connect(self.plotPressed)
        
        # Layout PLOT Bands
        self.lay_plot_band = self.findChild(QtWidgets.QVBoxLayout, "plotLayoutBand")
        self._canvas_band = FigureCanvas(Figure(figsize=(5,3)))
        self.lay_plot_band.addWidget(self._canvas_band)

        # TreeWidget orbitals bands
        self.tree_orbital = self.findChild(QtWidgets.QTreeWidget, "treeWidget_orbital")
        self.pb_chk_orb = self.findChild(QtWidgets.QPushButton, "pushButton_check_orbital")
        self.pb_chk_orb.clicked.connect(self.check_orbitalPressed)

        # Input Groupbox information.json BANDS
        self.pte_input_band = self.findChild(QtWidgets.QPlainTextEdit, "plainTextEdit_band")
        self.pb_browse_band = self.findChild(QtWidgets.QPushButton, "pushButton_browse_band")
        self.pb_browse_band.clicked.connect(self.browsePressed_band)

        # Energy Range Groupbox
        self.le_max_energy = self.findChild(QtWidgets.QLineEdit, "lineEdit_band_max")
        self.le_min_energy = self.findChild(QtWidgets.QLineEdit, "lineEdit_band_min")
        self.pb_plot_band = self.findChild(QtWidgets.QPushButton, "pushButton_plot_band")
        self.pb_plot_band.clicked.connect(self.plotBandPressed)

        self.show()


    def browsePressed_dos(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'open file', '~')
        if len(fname[0]) != 0:
            logging.info("Reading input file from : {}".format(fname[0]))
            self.pte_input_dos.clear()
            self.pte_input_dos.insertPlainText(fname[0])
            self.input_dos = fname[0]


    def plotPressed(self):
        start_freq = np.float64(self.le_start.text())
        stop_freq = np.float64(self.le_stop.text())
        intmax = np.float64(self.le_intmax.text())
        intmin = np.float64(self.le_intmin.text())
        logging.info("Frequency start : {}, stop : {}".format(start_freq,stop_freq))
        self._canvas_dos.figure.clear()
        ax = self._canvas_dos.figure.subplots()

        data = np.load(self.input_dos)
        freq = data["arr_0"]
        tdos = data["arr_1"]
        
        ax.set_xlim([start_freq, stop_freq])
        ax.set_ylim([intmin, intmax])
        ax.plot(freq, tdos)
        self._canvas_dos.figure.canvas.draw()


    def browsePressed_band(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'open file information.json', '~')
        if len(fname[0]) != 0:
            logging.info("Reading input information.json file from : {}".format(fname[0]))
            self.pte_input_band.clear()
            self.pte_input_band.insertPlainText(fname[0])
            self.var = jsonio(fname[0])
            self.var.summary_IO()

            self._show_tree_widget_orbitals()

            loc_hamiltonian_band = self.var.hamiltonian_bands
            num_orbitals = self.var.num_orbitals
            kpath = self.var.kpath
            self._num_kpath = self.var.num_kpath

            k_path = k_path_custom(k_point_selection = kpath, n = self._num_kpath)
            num_k_bands = len(k_path)
            self.hamiltonian_band = hamiltonian_band = np.memmap(
                filename = loc_hamiltonian_band,
                dtype = np.complex128, mode = 'r',
                shape = (num_k_bands, num_orbitals, num_orbitals)
            )

            self._eigen = np.linalg.eigvals(hamiltonian_band)
            self._eigen = np.sort(self._eigen, axis=1)
            logging.info("Eigenvalues has been calculated")


    def _show_tree_widget_orbitals(self):
        variable = self.var
        self.data = data = pd.read_csv(variable.orbital_index)
        for atom in data.atom.unique():
            parent = QtWidgets.QTreeWidgetItem(self.tree_orbital)
            parent.setText(0, "{}".format(atom))
            parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
            parent.setCheckState(0, Qt.Checked)
            filter_atom = data.atom == atom
            shells = data[filter_atom].n
            for shell in shells.unique():
                child = QtWidgets.QTreeWidgetItem(parent)
                child.setFlags(child.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                child.setText(0, "{}".format(shell))
                child.setCheckState(0, Qt.Checked)
                filter_shell = data[filter_atom].n == shell
                orbitals = data[filter_atom & filter_shell].orbital
                for orbital in orbitals:
                    grandchild = QtWidgets.QTreeWidgetItem(child)
                    grandchild.setFlags(grandchild.flags() | Qt.ItemIsUserCheckable)
                    grandchild.setText(0, "{}".format(orbital))
                    grandchild.setCheckState(0, Qt.Checked)


    def check_orbitalPressed(self):
        data = self.data
        unchecked = dict()
        root = self.tree_orbital.invisibleRootItem()
        parent_count = root.childCount()
        for i in range(parent_count):
            unchecked_atom = dict()
            parent = root.child(i)
            child_count = parent.childCount()
            for n in range(child_count):
                child = parent.child(n)
                sweeps = list()
                grandchild_count = child.childCount()
                for m in range(grandchild_count):
                    grandchild = child.child(m)
                    if grandchild.checkState(0) == QtCore.Qt.Unchecked:
                        sweeps.append(grandchild.text(0))
                        unchecked_atom[child.text(0)] = sweeps
                        unchecked[parent.text(0)] = unchecked_atom
        logging.debug(unchecked)

        idx_reduce = list()
        for atom in unchecked.keys():
            logging.debug("{}".format(atom))
            for shell in unchecked[atom].keys():
                logging.debug("    {}".format(shell))
                for orbital in unchecked[atom][shell]:
                    filter_atom = data.atom == atom
                    filter_shell = data.n == int(shell)
                    filter_orbital = data.orbital == orbital
                    idx = data[filter_atom & filter_shell & filter_orbital].orbital_index.values[0]
                    idx_reduce.append(idx-1)
                    logging.debug("        {}  {}".format(orbital, idx))
        logging.debug(idx_reduce)

        hamiltonian = self.hamiltonian_band
        ham_reduce = np.delete(hamiltonian, idx_reduce, axis=1)
        ham_reduce = np.delete(ham_reduce, idx_reduce, axis=2)
        logging.info("Hamiltonian shape : {}".format(ham_reduce.shape))

        logging.info("Calculating Eigenvalues .......")
        eigen_reduce = np.linalg.eigvals(ham_reduce)
        eigen_reduce = np.sort(eigen_reduce, axis=1)

        energy_max = np.float64(self.le_max_energy.text())
        energy_min = np.float64(self.le_min_energy.text())
        self._canvas_band.figure.clear()
        ax = self._canvas_band.figure.subplots()
        vline = [0]
        x = 0
        for i in self._num_kpath:
            x += i - 1
            vline.append(x)
        ax.plot(eigen_reduce, 'k-')
        ax.set_ylim([energy_min, energy_max])
        ax.set_xlim([0, x])
        for i in vline:
            ax.axvline(i, color='black')
        self._canvas_band.figure.canvas.draw()        
        logging.info("Finishing Calculate eigenvalues and show the band structures")


    def plotBandPressed(self):
        energy_max = np.float64(self.le_max_energy.text())
        energy_min = np.float64(self.le_min_energy.text())

        self._canvas_band.figure.clear()
        ax = self._canvas_band.figure.subplots()
        vline = [0]
        x = 0
        for i in self._num_kpath:
            x += i - 1
            vline.append(x)
        ax.plot(self._eigen, 'k-')
        ax.set_ylim([energy_min, energy_max])
        ax.set_xlim([0, x])
        for i in vline:
            ax.axvline(i, color='black')
        self._canvas_band.figure.canvas.draw()


if __name__ == "__main__" :
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()