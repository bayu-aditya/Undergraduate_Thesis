# Author : Bayu Aditya
from PyQt5 import QtWidgets, uic, QtWebEngineWidgets
from PyQt5.Qt import Qt
from PyQt5.QtCore import QEvent

import numpy as np
import pandas as pd
import logging
import time
import sys
import os
import json
import threading
import dash
import dash_core_components as dcc
import dash_html_components as html
sys.path.append('../')

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from src.variable import jsonio
from src.k_path import k_path_custom

class Ui(QtWidgets.QTabWidget):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi(uifile="mainwindow2.ui", baseinstance=self)
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

        # INPUT FILE TAB
        self.pb_browse_json = self.findChild(QtWidgets.QPushButton, "pb_browse_json")
        self.le_browse_json = self.findChild(QtWidgets.QLineEdit, "le_browse_json")
        self.pb_browse_json.clicked.connect(self.browsePressed_json)

        self.le_browse_hamband = self.findChild(QtWidgets.QLineEdit, "le_browse_hamband")
        self.le_browse_dos = self.findChild(QtWidgets.QLineEdit, "le_browse_dos")
        self.le_browse_orbital = self.findChild(QtWidgets.QLineEdit, "le_browse_orbital")
        self.le_browse_atompos = self.findChild(QtWidgets.QLineEdit, "le_browse_atompos")

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

        # Browse DOS Quantum Espresso
        self.pte_input_dosqe = self.findChild(QtWidgets.QPlainTextEdit, "plainTextEdit_dosqe")
        self.pb_browse_dosqe = self.findChild(QtWidgets.QPushButton, "pushButton_browse_dosqe")
        self.pb_browse_dosqe.clicked.connect(self.browsePressed_dosqe)

        # PDOS GroupBox
        self.tree_pdos = self.findChild(QtWidgets.QTreeWidget, "treeWidget_dos")
        self.pb_plot_pdos = self.findChild(QtWidgets.QPushButton, "pb_pdos")
        self.pb_plot_pdos.clicked.connect(self.plotPDOSpressed)
        self.sb_num_pdos = self.findChild(QtWidgets.QSpinBox, "sb_num_pdos")
        self.sb_num_pdos.valueChanged.connect(self.num_pdos_changed)

        # Layout PLOT Bands
        self.lay_plot_band = self.findChild(QtWidgets.QVBoxLayout, "plotLayoutBand")
        self._canvas_band = FigureCanvas(Figure(figsize=(5,3)))
        self.lay_plot_band.addWidget(self._canvas_band)

        # TreeWidget orbitals bands
        self.tree_orbital = self.findChild(QtWidgets.QTreeWidget, "treeWidget_orbital")
        self.pb_chk_orb = self.findChild(QtWidgets.QPushButton, "pushButton_check_orbital")
        self.pb_save_bands = self.findChild(QtWidgets.QPushButton, "pb_save_bands")
        self.pb_load_bands = self.findChild(QtWidgets.QPushButton, "pb_load_bands")
        self.pb_chk_orb.clicked.connect(self.check_orbitalPressed)
        self.pb_save_bands.clicked.connect(self.save_conf_bands)
        self.pb_load_bands.clicked.connect(self.load_conf_bands)

        # Input Groupbox information.json BANDS
        self.pte_input_band = self.findChild(QtWidgets.QPlainTextEdit, "plainTextEdit_band")
        self.pb_browse_band = self.findChild(QtWidgets.QPushButton, "pushButton_browse_band")
        self.pb_browse_band.clicked.connect(self.browsePressed_band)

        # Energy Range Groupbox
        self.le_max_energy = self.findChild(QtWidgets.QLineEdit, "lineEdit_band_max")
        self.le_min_energy = self.findChild(QtWidgets.QLineEdit, "lineEdit_band_min")
        self.pb_plot_band = self.findChild(QtWidgets.QPushButton, "pushButton_plot_band")
        self.pb_plot_band.clicked.connect(self.plotBandPressed)

        # Layout PLOT Atomic Position
        self.lay_plot_atompos = self.findChild(QtWidgets.QVBoxLayout, "plotLayoutAtomicPosition")
        # self._canvas_atompos = FigureCanvas(Figure(figsize=(5,5)))
        # self.lay_plot_atompos.addWidget(self._canvas_atompos)

        # Table Widget Atomic Position
        self.tb_atom_pos = self.findChild(QtWidgets.QTableWidget, "tb_atomic_position")

        self.show()


    def browsePressed_json(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, "open file information.json")
        if len(fname[0]) != 0:
            logging.info("Reading JSON file from : {}".format(fname[0]))
            self.le_browse_json.clear()
            self.le_browse_json.setText(fname[0])

            self.var = jsonio(fname[0])

            hamband_loc = self.var.hamiltonian_bands
            logging.info("Reading Hamiltonian Band Structures file from : {}".format(hamband_loc))
            self.le_browse_hamband.clear()
            self.le_browse_hamband.setText(hamband_loc)

            dos_loc = os.path.join(self.var.outdir, "DOS_array.npz")
            logging.info("Reading Density of States file from : {}".format(dos_loc))
            self.le_browse_dos.clear()
            self.le_browse_dos.setText(dos_loc)

            orbital_loc = self.var.orbital_index
            logging.info("Reading Orbital Index file from : {}".format(orbital_loc))
            self.le_browse_orbital.clear()
            self.le_browse_orbital.setText(orbital_loc)

            atompos_loc = self.var.atomic_position
            logging.info("Reading Atomic Positions  file from : {}".format(atompos_loc))
            self.le_browse_atompos.clear()
            self.le_browse_atompos.setText(atompos_loc)


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


    def browsePressed_dosqe(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'open file DOS QE', '*.dos')
        if len(fname[0]) != 0:
            logging.info("Reading input file from : {}".format(fname[0]))
            self.pte_input_dosqe.clear()
            self.pte_input_dosqe.insertPlainText(fname[0])
            input_dos_qe = fname[0]
            
            self._canvas_dos.figure.clear()
            data = pd.read_csv(input_dos_qe, skiprows=1, header=None, delim_whitespace=True)
            freq = data.iloc[:,0]
            dos = data.iloc[:,1]

            tools.plot_density_of_state(
                canvas = self._canvas_dos,
                dataX = freq, dataY = dos,
                le_start_freq = self.le_start, le_stop_freq = self.le_stop,
                le_intmin = self.le_intmin, le_intmax = self.le_intmax,
                label = "Quantum Espresso"
            )


    def num_pdos_changed(self):
        self.tree_pdos.clear()
        variable = self.var
        self.data = data = pd.read_csv(variable.orbital_index)

        num_lines = self.sb_num_pdos.value()
        for line in range(num_lines):
            ln = QtWidgets.QTreeWidgetItem(self.tree_pdos)
            ln.setText(0, "Line {}".format(line))
            tools.create_tree_orbitals(ln, data, "unchecked")


    def plotPDOSpressed(self):
        variable = self.var
        data_orbital_csv = self.data
        dos_loc = os.path.join(variable.outdir, "DOS_array.npz")
        
        start_freq = np.float64(self.le_start.text())
        stop_freq = np.float64(self.le_stop.text())
        dosmax = np.float64(self.le_intmax.text())
        dosmin = np.float64(self.le_intmin.text())

        dos = np.load(dos_loc)
        freq = dos["arr_0"]
        pdos = dos["arr_2"]

        line = self.tree_pdos.invisibleRootItem()
        line_count = line.childCount()
        dictionary = dict()
        for i in range(line_count):
            root = line.child(i)
            orbital_per_line = tools.check_tree_orbitals(root, data_orbital_csv, "checked")
            if len(orbital_per_line) != 0:
                dictionary[root.text(0)] = orbital_per_line
        logging.debug("dictionary pdos per-line : {}".format(dictionary))

        self._canvas_dos.figure.clear()
        ax = self._canvas_dos.figure.subplots()
        ax.set_xlim([start_freq, stop_freq])
        ax.set_ylim([dosmin, dosmax])
        for line in dictionary.keys():
            idx_reduce = dictionary[line]
            ax.plot(freq, pdos[:,idx_reduce].sum(axis=1), label=line)
        ax.legend()

        self._canvas_dos.figure.canvas.draw()


    def browsePressed_band(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'open file information.json', '~')
        if len(fname[0]) != 0:
            logging.info("Reading input information.json file from : {}".format(fname[0]))
            self.pte_input_band.clear()
            self.pte_input_band.insertPlainText(fname[0])
            self.var = jsonio(fname[0])
            self.var.summary_IO()

            self.tree_orbital.clear()
            self._show_tree_widget_orbitals()
            self._show_table_atomic_positions()
            self._show_atomic_positions()

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

            self._eigen = np.linalg.eigvalsh(hamiltonian_band)
            self._eigen = np.sort(self._eigen, axis=1)
            logging.info("Eigenvalues has been calculated")
            self.pb_plot_band.setEnabled(True)
            self.pb_chk_orb.setEnabled(True)


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


    def _show_table_atomic_positions(self):
        data = pd.read_csv(self.var.atomic_position)
        self.tb_atom_pos.setRowCount(len(data))
        self.tb_atom_pos.setColumnCount(len(data.columns))
        table = self.tb_atom_pos
        for i in range(len(data.atom.values)):
            for j in range(len(data.columns)):
                table.setItem(i,j, QtWidgets.QTableWidgetItem(
                    str(data.iloc[i,j]))
                    )


    def save_conf_bands(self):
        data_orbital_csv = self.data
        root = self.tree_orbital.invisibleRootItem()
        dict_orbital = tools.scan_tree_orbitals(root, data_orbital_csv, "unchecked")
        logging.debug(dict_orbital)
        fname = QtWidgets.QFileDialog.getSaveFileName(self, "save orbital reduction")
        if len(fname[0]) != 0:
            tools.dict_save_json(dict_orbital, fname[0])
            logging.info("Saving orbital reduction to : {}".format(fname[0]))


    def load_conf_bands(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'open file orbital reduction json', '~')
        if len(fname[0]) != 0:
            self.tree_orbital.clear()
            data = self.data
            uncheck_dict = tools.dict_load_json(fname[0])
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
                    shell = str(shell)
                    for orbital in orbitals:
                        grandchild = QtWidgets.QTreeWidgetItem(child)
                        grandchild.setFlags(grandchild.flags() | Qt.ItemIsUserCheckable)
                        grandchild.setText(0, "{}".format(orbital))
                        if atom in uncheck_dict.keys():
                            if shell in uncheck_dict[atom].keys():
                                if orbital in uncheck_dict[atom][shell]:
                                    grandchild.setCheckState(0, Qt.Unchecked)
                        else:
                            grandchild.setCheckState(0, Qt.Checked)
            logging.info("Load orbital reduction from : {}".format(fname[0]))


    def check_orbitalPressed(self):
        start = time.time()
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
        eigen_reduce = np.linalg.eigvalsh(ham_reduce)
        eigen_reduce = np.sort(eigen_reduce, axis=1)
        # eigen_reduce = eigen_multiprocessing(ham_reduce)

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
        logging.info("Finishing Calculate eigenvalues and show the band structures during {} sec.".format(time.time() - start))


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

    def _show_atomic_positions(self):
        pass
        # def run_dash(data, layout):
        #     app = dash.Dash()

        #     app.layout = html.Div(children=[
        #         html.H1(children='Hello Dash'),

        #         html.Div(children='''
        #             Dash: A web application framework for Python.
        #         '''),

        #         dcc.Graph(
        #             id='example-graph',
        #             figure={
        #                 'data': data,
        #                 'layout': layout
        #             })
        #         ])
        #     app.run_server(debug=False)
        
        # data = [
        # {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
        # {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
        # ]
        # layout = {
        #     'title': 'Dash Data Visualization'
        # }
        # threading.Thread(target=run_dash, args=(data, layout), daemon=True).start()


    # def _show_atomic_positions(self):
    #     data = pd.read_csv(self.var.atomic_position)

    #     self._canvas_atompos.figure.clear()
    #     ax = self._canvas_atompos.figure.add_subplot(1,1,1, projection='3d')

    #     x = data.posX.values*self.var.a
    #     y = data.posY.values*self.var.b
    #     z = data.posZ.values*self.var.c
    #     ax.set_xlim([0,60])
    #     ax.set_ylim([0,60])
    #     ax.set_zlim([0,60])
    #     ax.scatter(x,y,z)

    #     self._canvas_atompos.figure.canvas.draw()


class tools:
    @staticmethod
    def dict_save_json(dict_orbital, loc_json):
        with open(loc_json, 'w') as json_file:
            json.dump(dict_orbital, json_file, indent=4)


    @staticmethod
    def dict_load_json(loc_json):
        with open(loc_json, 'r') as read:
            dict_orbital = json.load(read)
        return dict_orbital


    @staticmethod
    def create_tree_orbitals(root_tree, dataframe_orbital_index, initial_state):
        root = root_tree
        data = dataframe_orbital_index
        state = Qt.Checked if initial_state == "checked" else Qt.Unchecked
        for atom in data.atom.unique():
            parent = QtWidgets.QTreeWidgetItem(root)
            parent.setText(0, "{}".format(atom))
            parent.setFlags(parent.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
            parent.setCheckState(0, state)
            filter_atom = data.atom == atom
            shells = data[filter_atom].n
            for shell in shells.unique():
                child = QtWidgets.QTreeWidgetItem(parent)
                child.setFlags(child.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable)
                child.setText(0, "{}".format(shell))
                child.setCheckState(0, state)
                filter_shell = data[filter_atom].n == shell
                orbitals = data[filter_atom & filter_shell].orbital
                for orbital in orbitals:
                    grandchild = QtWidgets.QTreeWidgetItem(child)
                    grandchild.setFlags(grandchild.flags() | Qt.ItemIsUserCheckable)
                    grandchild.setText(0, "{}".format(orbital))
                    grandchild.setCheckState(0, state)


    @staticmethod
    def scan_tree_orbitals(root_tree, dataframe_orbital, check_state):
        parent_count = root_tree.childCount()
        state = Qt.Checked if check_state == "checked" else Qt.Unchecked

        # create dictionary orbital
        dictionary = dict()
        for i in range(parent_count):
            dict_atom = dict()
            parent = root_tree.child(i)
            child_count = parent.childCount()
            for n in range(child_count):
                child = parent.child(n)
                sweeps = list()
                grandchild_count = child.childCount()
                for m in range(grandchild_count):
                    grandchild = child.child(m)
                    if grandchild.checkState(0) == state:
                        sweeps.append(grandchild.text(0))
                        dict_atom[child.text(0)] = sweeps
                        dictionary[parent.text(0)] = dict_atom
        return dictionary


    @staticmethod
    def check_tree_orbitals(root_tree, dataframe_orbital, check_state):
        parent_count = root_tree.childCount()
        data = dataframe_orbital
        state = Qt.Checked if check_state == "checked" else Qt.Unchecked

        # create dictionary orbital
        dictionary = dict()
        for i in range(parent_count):
            dict_atom = dict()
            parent = root_tree.child(i)
            child_count = parent.childCount()
            for n in range(child_count):
                child = parent.child(n)
                sweeps = list()
                grandchild_count = child.childCount()
                for m in range(grandchild_count):
                    grandchild = child.child(m)
                    if grandchild.checkState(0) == state:
                        sweeps.append(grandchild.text(0))
                        dict_atom[child.text(0)] = sweeps
                        dictionary[parent.text(0)] = dict_atom
        logging.debug(dictionary)
        
        # dictionary orbital to index orbital
        idx_orbital = list()
        for atom in dictionary.keys():
            logging.debug("{}".format(atom))
            for shell in dictionary[atom].keys():
                logging.debug("     {}".format(shell))
                for orbital in dictionary[atom][shell]:
                    filter_atom = data.atom == atom
                    filter_shell = data.n == int(shell)
                    filter_orbital = data.orbital == orbital
                    idx = data[filter_atom & filter_shell & filter_orbital].orbital_index.values[0]
                    idx_orbital.append(idx-1)
                    logging.debug("        {}  {}".format(orbital, idx))
        logging.debug(idx_orbital)
        return idx_orbital


    @staticmethod
    def plot_density_of_state(
        dataX, dataY, canvas, le_start_freq, le_stop_freq, le_intmin, le_intmax, label):
        start_freq = np.float64(le_start_freq.text())
        stop_freq = np.float64(le_stop_freq.text())
        intmax = np.float64(le_intmax.text())
        intmin = np.float64(le_intmin.text())

        ax = canvas.figure.subplots()
        ax.set_xlim([start_freq, stop_freq])
        ax.set_ylim([intmin, intmax])
        ax.plot(dataX, dataY, label=label)
        ax.legend()
        canvas.figure.canvas.draw()


if __name__ == "__main__" :
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    app.exec_()