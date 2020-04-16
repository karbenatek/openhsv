from PyQt5.QtWidgets import QWidget, QGridLayout
from skimage.util import pad
import pyqtgraph as pg
from skimage.color import rgb2gray
from tensorflow.keras.models import load_model
import numpy as np


def divpad(im, multiple_of=32, cval=0):
    """Padding of images that each dimension is a multiple of a given number.

    Parameters
    ----------
    im : numpy.ndarray
        The input image
    multiple_of : int, optional
        the desired multiple of in each dimension, by default 32
    cval : int, optional
        the padding value, by default 0

    Returns
    -------
    numpy.ndarray
        The padded image
    """
    needed_padding = []
    real_padding = []

    for sh in im.shape:
        if sh > 3 and sh % multiple_of:
            needed_padding.append(multiple_of - sh % multiple_of)
        else:
            needed_padding.append(0)

    real_padding.append([needed_padding[0] // 2,
                         needed_padding[0] // 2 + needed_padding[0] % 2])

    real_padding.append([needed_padding[1] // 2,
                         needed_padding[1] // 2 + needed_padding[1] % 2])

    return pad(im, real_padding, 'constant', constant_values=cval)

class Analysis(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Analysis - Full automatic glottis segmentation")

        self.model = None
        self.segmentations = []
        self.GAW = []

        self.initUI()
        self.initTensorflow()

    def initUI(self):
        self.l = QGridLayout()
        self.setLayout(self.l)

        self.setGeometry(50, 50, 1800, 600)

        pen = pg.mkPen("y", width=3)

        self.im = pg.ImageView()
        self.seg = pg.ImageView()
        self.plt = pg.PlotWidget()
        self.plt.setMaximumWidth(400)
        self.curve = self.plt.plot(pen=pen, symbolBrush=(255, 255, 255), symbolPen="w", symbolSize=8, symbol="o")

        self.im.setImage(np.random.randint(0, 100, (200, 200)))
        self.seg.setImage(np.random.randint(0, 100, (200, 200)))

        self.l.addWidget(self.im, 0, 0, 2, 1)
        self.l.addWidget(self.seg, 0, 2, 2, 1)
        self.l.addWidget(self.plt, 0, 4, 1, 1)

    def initTensorflow(self):
        self.model = load_model(r"C:\Users\kistas\PycharmProjects\idthse\GlottisSegmentation.h5", compile=False)

    def segment(self, im):
        """
        Segments an endoscopic image using a deep neural network
        :param im: np.ndarray (HxWx3)
        :return:
        """
        # Process image to fit the DNN
        processed = divpad(rgb2gray(im).astype(np.float32))
        # print(processed.min(), processed.max())
        # processed = processed * 2 - 1

        # Run neural network
        pr = self.model.predict(processed[None, ..., None]).squeeze()

        # Save segmentation and GAW
        self.segmentations.append(pr)
        self.GAW.append(pr.sum())

        # Transpose image if RGB
        if im.ndim == 3:
            im = im.transpose((1, 0, 2))

        # Transpose image if grayscale
        if im.ndim == 2:
            im = im.transpose((1, 0))

        # Show image, show segmentation, show GAW
        self.im.setImage(im)
        self.seg.setImage(pr.transpose((1, 0)))
        self.curve.setData(self.GAW)

    def get(self):
        return dict(gaw=self.GAW, segmentation=self.segmentations)

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import imageio as io

    app = QApplication([])
    a = Analysis()
    vid = io.mimread(r"anki_vocal_folds_small.mp4", memtest=False)
    a.show()

    for i in range(len(vid)):
        a.segment(vid[i].astype(np.float32)/127-1)
        app.processEvents()

    app.exec_()