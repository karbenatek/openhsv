from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox, QWidget
import pyqtgraph as pg
import numpy as np
import imageio as io
import flammkuchen as fl

if __name__ == '__main__':
    app = QApplication([])

    fn = QFileDialog.getOpenFileName(filter="*.mp4; *.audio; *.raw")[0]

    if fn:
        if fn.endswith("audio"):
            x = fl.load(fn)['audio']
            pg.plot(x[:, 0])
            pg.plot(x[:, 1])

        elif fn.endswith("raw"):
            x = fl.load(fn)['ims']

            if x.ndim == 3:
                x = x[..., None]
            pg.image(x.transpose(0, 2, 1, 3))

        elif fn.endswith("mp4"):
            x = io.mimread(fn, memtest=False)
            pg.image(np.asarray(x, dtype=np.uint8).transpose(0,2,1,3))

        else:
            QMessageBox.information(QWidget(), "Could not open file",
                                    "File {} could not be opened!".format(fn))

    app.exec_()