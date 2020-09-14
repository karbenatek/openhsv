from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel
from skimage.util import pad
import pyqtgraph as pg
from skimage.color import rgb2gray
import numpy as np
from tqdm import tqdm
from openhsv.analysis.midline import Midline
from openhsv.analysis.parameters import GAW
from openhsv.gui.table import Table

def _divpad(im, multiple_of=32, cval=0):
    """preprocesses an cropped image for feeding into neural network.
    In most convolutional neural networks, images need to have a specific minimum
    size that it can be processed by the network. In a U-Net-like architecture,
    image dimensions should be a multiple of 32.
    
    :param im: cropped input image (grayscale or RGB)
    :type im: numpy.ndarray
    :param multiple_of: number image dimensions should be a multiple of, defaults to 32
    :type multiple_of: int, optional
    :param cval: value that should be used for padded columns, defaults to 0
    :type cval: int, optional
    :return: padded input image
    :rtype: numpy.ndarray
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
    def __init__(self, app=None):
        """Analysis widget that shows the segmentation process of the neural network.
        
        :param QWidget: Inherits from QWidget
        :type QWidget: PyQt5.QtWidgets.QWidget
        :param app: QApplication, needed to process events to avoid freezing of the GUI, defaults to None
        :type app: PyQt5.QtWidgets.QWidget, optional
        """
        super().__init__()

        self.setWindowTitle("Analysis - Full automatic glottis segmentation")

        self.app = app
        self.model = None
        self.segmentations = []
        self.GAW = []

        self.initUI()
        self.initTensorflow()

    def initUI(self):
        """inits the user interface. In particular, it prepares the preview window for
        the endoscopic image, the segmentation map and the glottal area waveform (GAW).
        """
        self.l = QGridLayout(self)
        self.setGeometry(50, 50, 1800, 600)

        pen = pg.mkPen("y", width=2)

        # Preview Endoscopic image
        self.im = pg.ImageView()
        # Preview segmentation
        self.seg = pg.ImageView()
        # Preview GAW
        self.plt = pg.PlotWidget()
        self.plt.setMaximumWidth(400)
        self.curve = self.plt.plot(pen=pen, symbolBrush=(255, 255, 255), symbolPen="w", symbolSize=8, symbol="o")

        # Set dummy image - needed?!
        self.im.setImage(np.random.randint(0, 100, (200, 200)))
        self.seg.setImage(np.random.randint(0, 100, (200, 200)))

        self.l.addWidget(QLabel("Endoscopy image"), 0, 0, 1, 1)
        self.l.addWidget(QLabel("Segmentation map"), 0, 2, 1, 1)
        self.l.addWidget(QLabel("Glottal area waveform (GAW)"), 0, 4, 1, 1)

        self.l.addWidget(self.im, 1, 0, 1, 1)
        self.l.addWidget(self.seg, 1, 2, 1, 1)
        self.l.addWidget(self.plt, 1, 4, 1, 1)

    def initTensorflow(self):
        """Initializes tensorflow and loads glottis segmentation neural network
        """
        from tensorflow.keras.models import load_model
        self.model = load_model(r"./openhsv/cnn/GlottisSegmentation.h5", compile=False)

    def segmentSequence(self, ims, normalize=True, reinit=True):
        """segments an image sequence, such as a video, frame by frame.
        
        :param ims: collection of images
        :type ims: list of numpy.ndarray, or numpy.ndarray
        :param normalize: normalize 0..255 to -1..1, defaults to True
        :type normalize: bool, optional
        :param reinit: deletes any previous segmentation information, defaults to True
        :type reinit: bool, optional
        """
        if reinit:
            self.GAW = []
            self.segmentations = []

        for im in tqdm(ims):
            if normalize:
                im = im.astype(np.float32) / 127.5 - 1

            # Segment frame
            self.segment(im)

            # Ensure that the GUI is response
            if self.app:
                app.processEvents()

    def segment(self, im):
        """Segments an endoscopic image using a deep neural network

        :param im: np.ndarray (HxWx3)
        :return:
        """
        # Process image to fit the DNN
        processed = _divpad(rgb2gray(im).astype(np.float32))
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
        elif im.ndim == 2:
            im = im.transpose((1, 0))

        # Show image, show segmentation, show GAW
        self.im.setImage(im)
        self.seg.setImage(pr.transpose((1, 0)))
        self.curve.setData(self.GAW[-40:])

    def computeParameters(self, dt=1/4000, debug=False):
        """Compute parameters from GAW

        :param dt: sampling time in seconds, defaults to 1/4000
        :type dt: float, optional
        :param debug: shows debugging information and plots, defaults to False
        :type debug: bool, optional
        """
        # Convert raw segmentations to binary masks
        seg = np.asarray(self.segmentations).round().astype(np.bool)

        # Predict midline from segmentation
        M = Midline(seg)
        M.predict()

        # Use midline for left and right GAW
        gaws = M.side()
        left_gaw  = gaws[..., 0]
        right_gaw = gaws[..., 1]

        # Compute and show values
        gaw = GAW(seg.sum((1,2)), 
            use_filtered_signal=False, 
            use_hanning=False, 
            dt=dt)

        gaw.setLeftRightGAW(left_gaw, right_gaw)
        params = gaw.computeParameters()

        # Create summary table for parameters
        self.t = Table(params)
        self.t.show()

        if debug:
            # Show complete segmentation with midline
            im = pg.image(seg.transpose(0, 2, 1), 
                        title="Segmentation with midline")

            line = pg.LineSegmentROI([M.coordinates[0, :2],
                                    M.coordinates[0, 2:],],
                                    pen="y")

            im.getView().addItem(line)

            # Show complete GAW plot with detected cycles
            gaw_plot = pg.plot(gaw.t, gaw.raw_signal,
                title="GAW with cycles")

            cs = [(241, 196, 15), (231, 76, 60)]
            i = 0

            for o, c in zip(gaw.opening, gaw.closing):
                i1 = pg.PlotCurveItem(gaw.t[o:c], np.zeros_like(gaw.t[o:c]))
                i2 = pg.PlotCurveItem(gaw.t[o:c], gaw.raw_signal[o:c])
                between = pg.FillBetweenItem(i1, i2, brush=cs[i % len(cs)])
                gaw_plot.getPlotItem().addItem(between)
                i += 1

            # Show left and right gaw
            LR_plot = pg.plot(title="Left and right GAW")
            LR_plot.plot(gaw.t, left_gaw)
            LR_plot.plot(gaw.t, -right_gaw)

        # Compute and show phonovibrogram
        pvg = M.pvg()
        pg.image(pvg, title="Phonovibrogram")

        return params

    def get(self):
        """returns GAW and segmentation maps for video
        
        :return: GAW and segmentations
        :rtype: tuple(list, list(numpy.ndarray))
        """
        return dict(gaw=self.GAW, segmentation=self.segmentations)

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import imageio as io

    app = QApplication([])
    
    # Load an example video
    vid = io.mimread(r"./openhsv/examples/oscillating_vocal_folds.mp4",
        memtest=False)

    # Create analysis class and show widget
    a = Analysis(app)
    a.show()
    a.segmentSequence(vid)
    a.computeParameters(dt=1/1000, debug=True)

    app.exec_()