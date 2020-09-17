from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, \
    QGridLayout, QLineEdit, QTableWidget, QDateEdit, QTableWidgetItem, \
    QHeaderView
from PyQt5.QtCore import Qt
import pyqtgraph as pg 
from glob import glob
import imageio as io
import numpy as np
import json


class Table(QWidget):
    def __init__(self, folder):
        super().__init__()
        self.l = QGridLayout(self)
        self.setMinimumWidth(700)

        self.surname = QLineEdit()
        self.surname.setPlaceholderText("Last name, e.g. Smith")
        self.surname.textChanged.connect(self.search)
        self.firstname = QLineEdit()
        self.firstname.setPlaceholderText("First name")
        self.firstname.textChanged.connect(self.search)
        self.birthdate = QDateEdit()
        self.birthdate.dateChanged.connect(self.search)
        self.search = QPushButton("Search")

        self.l.addWidget(self.surname, 0, 0)
        self.l.addWidget(self.firstname, 0, 1)
        self.l.addWidget(self.birthdate, 0, 2)
        self.l.addWidget(self.search, 0, 3)

        self.folder = folder
        self.im = None

        self.t = QTableWidget()
        h = self.t.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.Stretch)
        

        self.fns = glob(folder+"\\**\\*.meta", recursive=True)
        self.patients = []
        self.opened = ""

        # Iterate over metadata files
        for ii, fn in enumerate(self.fns):
            meta = json.load(open(fn))

            meta['Patient']['date'] = meta['Date']

            # Store patient metadata
            self.patients.append(meta['Patient'])

            rc = self.t.rowCount()
            self.t.setRowCount(rc+1)

            # Set column when first file is loaded
            if ii == 0:
                self.t.setColumnCount(len(self.patients[0].keys()))
                self.t.setHorizontalHeaderLabels(self.patients[0].keys())

            # Show metadata in a single row
            for i, (k, v) in enumerate(self.patients[-1].items()):
                item = QTableWidgetItem(v)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.t.setItem(rc, i, item)

        # connect 
        self.t.doubleClicked.connect(self.do)

        self.l.addWidget(self.t, 1, 0, 4, 10)

    def search(self):
        surname = self.surname.text()
        firstname = self.firstname.text()
        birthdate = self.birthdate.text()

        # Iterate over rows
        # Only show rows that contain relevant data
        # As entered/selected in search
        for i in range(self.t.rowCount()):
            self.t.showRow(i)

            if surname:
                if surname not in self.t.item(i, 0).text():
                    self.t.hideRow(i)

            if firstname:
                if firstname not in self.t.item(i, 1).text():
                    self.t.hideRow(i)

            if birthdate != QDateEdit().text():
                if birthdate != self.t.item(i, 3).text():
                    self.t.hideRow(i)

            # if surname not in self.t.item(i, 0).text() or \
            #     firstname not in self.t.item(i, 1).text() or \
            #     birthdate != self.t.item(i, 3).text():
            #     self.t.hideRow(i)

            # if not firstname and not surname and birthdate == QDateEdit().text():
            #     self.t.showRow(i)

    def do(self, e):
        # Get clicked row
        row = e.row()

        # Get respective file name and look for video
        fn = self.fns[row]
        vid_fn = fn.replace(".meta", ".mp4")

        # If video is already opened, do nothing
        if vid_fn == self.opened and type(self.im) != None:
            if self.im.isVisible():
                return
        
        # Otherwise, read video
        vid = io.mimread(vid_fn,
            memtest=False)

        # Show video
        self.im = pg.image(np.asarray(vid, np.uint8).transpose(0,2,1,3),
            title=vid_fn)

        self.opened = vid_fn

        






class DB(QMainWindow):
    def __init__(self, folder):
        super().__init__()
        self.l = QGridLayout(self)
        self.t = Table(folder)
        self.folder = folder

        self.setCentralWidget(
            self.t
        )

        self.setWindowTitle("Patient Database Viewer")



if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication([])

    db = DB("C:/openhsv")
    db.show()

    sys.exit(app.exec_())