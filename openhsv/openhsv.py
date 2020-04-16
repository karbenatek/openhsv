# Libraries used
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QSlider, QPushButton, QProgressBar, QLabel, \
    QMessageBox, QSplashScreen, QSizePolicy, QDialog, QLineEdit, QDateEdit, QCheckBox, QComboBox, QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QColor, QPen, QIntValidator
from PyQt5.QtCore import Qt, QTimer
import qdarkstyle
import pyqtgraph as pg
import sys
import numpy as np
import imageio as io
import flammkuchen as fl  # Saving HDF5 data
import time
import sounddevice as sd  # Recording audio
import queue  # Saving audio in memory
import XsCamera
from threading import Timer
from glob import glob
from os.path import isdir
import os
from datetime import datetime

# Own scripts
from analysis import Analysis
from camera import Camera

"""
    pyqtgraph: change __init__ cleanup RuntimeError to ReferenceError (issue with tensorflow...)
    Needs better documentation throughout
    Split classes into separate files
    Re-program in C++
"""

class Settings(QDialog):
    def __init__(self, exposure, fps, audioSamplingRate, audioBlockSize, audioBufferSize, save_raw, base_folder):
        super().__init__()
        self.l = QGridLayout(self)
        self.onlyInt = QIntValidator()

        self.l.addWidget(QLabel("Exposure time [us], e.g. 245 for 4000 fps"))
        self.exposureTime = QLineEdit(str(exposure))
        self.exposureTime.setValidator(self.onlyInt)
        self.l.addWidget(self.exposureTime)

        self.l.addWidget(QLabel("Frames per second, e.g. 4000 for full frame"))
        self.fps = QLineEdit(str(fps))
        self.fps.setValidator(self.onlyInt)
        self.l.addWidget(self.fps)

        self.l.addWidget(QLabel("Audio Sample Rate, e.g. 96000 for 4000 camera fps"))
        self.audioSamplingRate = QLineEdit(str(audioSamplingRate))
        self.audioSamplingRate.setValidator(self.onlyInt)
        self.l.addWidget(self.audioSamplingRate)

        self.l.addWidget(QLabel("Audio Block Size, e.g. 4800"))
        self.audioBlockSize = QLineEdit(str(audioBlockSize))
        self.audioBlockSize.setValidator(self.onlyInt)
        self.l.addWidget(self.audioBlockSize)

        self.l.addWidget(QLabel("Audio Buffer Size, e.g. 8"))
        self.audioBufferSize = QLineEdit(str(audioBufferSize))
        self.audioBufferSize.setValidator(self.onlyInt)
        self.l.addWidget(self.audioBufferSize)

        self.l.addWidget(QLabel("Saving options"))
        self.save_raw = QCheckBox("Save also raw data")
        self.save_raw.setChecked(save_raw)
        self.l.addWidget(self.save_raw)

        self.l.addWidget(QLabel("Base folder:"))
        self.base_folder = QLabel(base_folder)
        self.select_base_folder = QPushButton("Select other base folder")
        self.select_base_folder.clicked.connect(self.selectBaseFolder)
        self.l.addWidget(self.base_folder)
        self.l.addWidget(self.select_base_folder)
        self.save_raw.setChecked(save_raw)
        self.l.addWidget(self.save_raw)

        b = QPushButton("Save settings")
        b.clicked.connect(self.close)

        self.l.addWidget(b)

        self.setWindowTitle("Settings")

    def selectBaseFolder(self):
        folder = QFileDialog.getExistingDirectory(caption="Select base folder")

        if folder:
            self.base_folder.setText(folder)

    def get(self):
        return dict(exposureTime=int(self.exposureTime.text()),
                    videoSamplingRate=int(self.fps.text()),
                    audioSamplingRate=int(self.audioSamplingRate.text()),
                    audioBlockSize=int(self.audioBlockSize.text()),
                    audioBufferSize=int(self.audioBufferSize.text()))


class Patient(QDialog):
    def __init__(self, base_folder=r"C:\openHSE"):
        super().__init__()

        self.setWindowTitle("New patient")
        self.setFixedWidth(300)

        self.l = QGridLayout(self)

        self.combo = QComboBox()

        folders = [i.split("\\")[-1] for i in glob(base_folder+"\\*") if isdir(i)]

        if len(folders) == 0:
            os.mkdir(base_folder+"\\Sprechstunde")
            folders.append("Sprechstunde")

        self.combo.addItem("Sprechstunde")

        for i in folders:
            if i is not "Sprechstunde":
                self.combo.addItem(i)

        self.l.addWidget(QLabel("Folder"))
        self.l.addWidget(self.combo)

        self.l.addWidget(QLabel('Last Name / Identifier'))
        self.last_name = QLineEdit()
        self.last_name.setPlaceholderText("e.g. Smith")
        self.l.addWidget(self.last_name)

        self.l.addWidget(QLabel("First Name"))
        self.first_name = QLineEdit()
        self.first_name.setPlaceholderText("e.g. John")
        self.l.addWidget(self.first_name)

        self.l.addWidget(QLabel("Birth data"))
        self.birth_date = QDateEdit()
        self.l.addWidget(self.birth_date)

        self.l.addWidget(QLabel("Comment"))
        self.comment = QLineEdit()
        self.comment.setPlaceholderText("e.g. RBH 022")
        self.l.addWidget(self.comment)

        b = QPushButton("Continue")
        b.clicked.connect(self.close)

        self.l.addWidget(b)

    def close(self):
        if not self.last_name.text():
            QMessageBox.information(self, "No identifier", "Please enter identifier (e.g. last name) of the subject.")
            return

        super().close()

    def get(self):
        return dict(last_name=self.last_name.text(),
                    first_name=self.first_name.text(),
                    birth_date=self.birth_date.text(),
                    comment=self.comment.text(),
                    folder=self.combo.currentText())

class IDTHSE (QWidget):
    def __init__(self, app):
        super().__init__()

        self.cam = None
        self.app = app
        self.play = False
        self.audioBlockSize = 5000
        self.audioSamplingRate = 50000
        self.buffer_id = 0
        self.audioBufferSize = 5
        self.audioBuffer = [0]*self.audioBufferSize
        self.hann = np.hanning(self.audioBlockSize*self.audioBufferSize)
        self.audioQueue = queue.Queue()
        self.t = QTimer()
        self.t.timeout.connect(self.nextFrame)

        self.abortSaving = False
        self.save_raw = True

        self.base_folder = r"C:\openHSE"

        # self.exposureTime = exposureTime
        # self.videoSamplingRate = videoSamplingRate
        # self.audioSamplingRate = audioSamplingRate
        # self.audioBlockSize = audioBlockSize
        # self.audioBufferSize = audioBufferSize

        self.triggerFrameIndex = 0
        self.cur_frame = 0

        # No analysis has been performed here...
        self.analysis = None

        self.setWindowTitle("openHSE v.0.4")
        self.setGeometry(100, 100, 800, 800)

        # Setup layout
        self.l = QGridLayout()
        self.setLayout(self.l)

        # Create camera / frame preview window with rectangle for image analysis
        self.im = pg.ImageView()
        self.im.setFixedWidth(1200)
        self.im.setImage(np.zeros((1000, 1000)))
        self.roi = pg.RectROI([0, 0], [100, 100])

        self.im.getView().addItem(self.roi)
        self.im.getView().setMenuEnabled(False)


        # Create audio preview window
        self.audio = pg.PlotWidget()
        self.audioData = []
        self.audio.setMaximumHeight(350)
        self.audioCurve1 = self.audio.plot(np.ones(1000,), pen=pg.mkPen('m'))
        self.audioCurve2 = self.audio.plot(np.ones(1000, )-2, pen=pg.mkPen('y'))

        # F0
        self.f0_item = pg.TextItem("x Hz")
        self.audio.addItem(self.f0_item)
        self.F0_timer = QTimer()
        self.F0_timer.timeout.connect(self.F0)

        # Imaging data
        self.imagingData = []

        # Add Widgets to Layout
        icon = QLabel()
        icon_pix = QPixmap("openhse_logo-01-01.png")
        icon.setFixedHeight(50)
        icon.setPixmap(icon_pix.scaled(160, 50, Qt.KeepAspectRatio))
        self.l.addWidget(icon, 0, 0, 1, 1)

        self.b3 = QPushButton("New patient")
        self.b3.clicked.connect(self.patient)
        self.b3.setFixedWidth(120)
        self.l.addWidget(self.b3, 1, 0)

        self.b4 = QPushButton("Change settings")
        self.b4.clicked.connect(self.settings)
        self.b4.setFixedWidth(120)
        self.l.addWidget(self.b4, 2, 0)

        LabelCamera = QLabel("Camera")
        LabelCamera.setFixedHeight(30)
        LabelAudio = QLabel("Audio")
        LabelAudio.setFixedHeight(30)
        self.l.addWidget(LabelCamera, 3, 0, 1, 1)
        self.l.addWidget(self.im, 4, 0, 10, 1)
        self.l.addWidget(LabelAudio, 3, 1, 1, 1)
        self.l.addWidget(self.audio, 4, 1, 5, 1)

        self.b = QPushButton("Start Camera Feed")
        self.b.clicked.connect(self.startCamera)

        self.b5 = QPushButton("Take screenshot")
        self.b5.clicked.connect(self.screenshot)


        # self.sl = QWidget()
        # self.sll = QHBoxLayout(self.sl)
        # self.sll.setContentsMargins(0, 0, 100, 100)

        self.start_slider = QSlider(orientation=Qt.Horizontal)
        self.start_slider.setMinimum(1)
        self.start_slider.setMaximum(4000)
        self.start_slider.setAutoFillBackground(False)
        self.start_slider.setStyleSheet(
            "QSlider::groove:horizontal, QSlider::groove:horizontal:hover, QSlider::sub-page:horizontal, QSlider::groove:horizontal:disabled { border:0;  background: #19232D; }")

        self.start_slider.valueChanged.connect(self.getFrameStart)

        # self.sll.addWidget(self.start_slider)

        self.end_slider = QSlider(orientation=Qt.Horizontal)
        self.end_slider.setMinimum(1)
        self.end_slider.setMaximum(4000)
        self.end_slider.setValue(4000)
        self.end_slider.setAutoFillBackground(False)
        self.end_slider.setStyleSheet("QSlider::groove:horizontal, QSlider::groove:horizontal:hover, QSlider::sub-page:horizontal, QSlider::groove:horizontal:disabled { border:0;  background: #19232D; }")
        self.end_slider.valueChanged.connect(self.getFrameEnd)

        self.l.addWidget(self.b, 7, 1 )
        self.l.addWidget(self.b5, 8, 1)
        # self.l.addWidget(QLabel("Start"))
        self.l.addWidget(self.start_slider, 20, 0, 1, 2)


        self.rangeIndicator = QLabel()
        # self.rangeIndicator.setContentsMargins(0, 0, 0, 0)
        # self.rangeIndicator.setMaximumHeight(20)
        self.rangeIndicator.setFixedHeight(25)
        self.rangeIndicator.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)


        self.l.addWidget(self.rangeIndicator, 21, 0, 1, 2)
        # self.l.addWidget(QLabel("End"))
        self.l.addWidget(self.end_slider, 22, 0, 1, 2)

        self.b1 = QPushButton("Start Analysis - Glottis Segmentation")
        self.b1.setEnabled(False)
        self.b1.clicked.connect(self.analyze)

        self.progess = QProgressBar()
        self.progess.setMinimum(0)
        self.progess.setMaximum(100)
        self.progess.setEnabled(False)

        self.l.addWidget(self.b1, 9, 1)
        self.l.addWidget(self.progess, 10, 1)

        self.saveButton = QPushButton("Save data.")
        self.saveButton.clicked.connect(self.save)
        self.saveButton.setEnabled(False)
        self.l.addWidget(self.saveButton, 11, 1)

        self.b2 = QPushButton("Play/Stop")
        self.b2.clicked.connect(self.playStop)
        self.b2.setEnabled(False)
        # self.l.addWidget(self.b2, 12, 1)

        self.updateRangeIndicator()
        self.initSettings()

    def screenshot(self):
        scr = self.im.getImageItem()

        im = scr.image
        io.imsave("Screenshot_{}.png".format(datetime.now().strftime("%Y%m%d_%H%M%S")), im)

        pass

    def settings(self):
        s = Settings(self.exposureTime,
                     self.videoSamplingRate,
                     self.audioSamplingRate,
                     self.audioBlockSize,
                     self.audioBufferSize,
                     self.save_raw,
                     self.base_folder)
        s.exec_()

        self.save_raw = s.save_raw.isChecked()
        self.base_folder = s.base_folder.text()
        self.initSettings(**s.get())

        if self.cam is not None:
            self.cam.closeCamera()
            self.initCamera(True)

    def initSettings(self,
                     exposureTime=245,
                     videoSamplingRate=4000,
                     audioSamplingRate=80000,
                     audioBlockSize=4000,
                     audioBufferSize=3):

        self.exposureTime = exposureTime
        self.videoSamplingRate = videoSamplingRate
        self.audioSamplingRate = audioSamplingRate
        self.audioBlockSize = audioBlockSize
        self.audioBufferSize = audioBufferSize

        self.audioBuffer = [0] * self.audioBufferSize
        self.hann = np.hanning(self.audioBlockSize * self.audioBufferSize)

    def patient(self):
        p = Patient(self.base_folder)
        p.exec_()

        self.patientInformation = p.get()

    def updateRangeIndicator(self):
        w, h = self.rangeIndicator.width(), self.rangeIndicator.height()

        pix = QPixmap(w, h)
        p = QPainter(pix)
        p.setBrush(QBrush(QColor("#19232D")))
        p.drawRect(-1, -1, w, h)
        p.setBrush(QBrush(QColor("#1464A0")))

        x0 = w*self.start_slider.value()/self.start_slider.maximum()
        x1 = w*self.end_slider.value()/self.end_slider.maximum()

        p.drawRect(int(x0), 0, int(x1-x0), h)
        p.setPen(QPen(QColor("#FFFFFF")))
        p.drawText(0, 17, "{} frames selected".format(self.end_slider.value()-self.start_slider.value()+1))
        p.end()

        self.rangeIndicator.setPixmap(pix)

    def crop(self, im):
        pos = self.roi.pos()
        size = self.roi.size()

        return im[int(pos[1]):int(pos[1]+size[1]), int(pos[0]):int(pos[0]+size[0])]

    def initCamera(self, force_init=False):
        if self.cam is None or force_init:
            self.cam = Camera()
            self.cam.setSettings(self.exposureTime, self.videoSamplingRate)
            self.cam.getBufferSize()

            self.saveButton.setEnabled(True)
            self.b1.setEnabled(True)
            self.b2.setEnabled(True)

    def playStop(self):
        if self.play:
            self.t.stop()
            self.play = False

        else:
            self.t.start(10)
            self.play = True

    def setImage(self, im):
        # Save view from current image
        state, levels = self.im.getView().getState(), self.im.getImageItem().levels
        # Set new image
        self.im.setImage(im)
        # Restore view
        self.im.getView().setState(state)
        # Restore levels
        # self.im.getImageItem().setLevels(levels)

    def nextFrame(self):
        self.cur_frame += 1

        if self.cur_frame >= self.cam.frames_to_record:
            self.cur_frame = 0

        im = self.cam.getMemoryFrame(self.cur_frame, by_trigger=True)
        self.setImage(im.transpose((1, 0, 2)))

    def initAudio(self):
        self.audioQueue = queue.Queue()
        self.audioData = []
        self.recorder = sd.InputStream(samplerate=self.audioSamplingRate, device=1, channels=2, callback=self.audioCallback, blocksize=self.audioBlockSize)
        self.recorder.start()

    def audioCallback(self, data, *args):
        self.audioCurve1.setData(data[:, 0]+1)
        self.audioCurve2.setData(data[:, 1]-1)

        self.audioQueue.put(data.copy())

    def stopAudio(self):
        self.recorder.stop()

        while not self.audioQueue.empty():
            self.F0()

    def F0(self):

        if self.audioQueue.empty():
            return

        data = self.audioQueue.get()
        self.audioData.append(data)
        d = data[:, 1]

        # self.buffer_id += 1
        # self.audioBuffer[self.audioBlockSize * (self.buffer_id % self.audioBufferSize):self.audioBlockSize * (
        #         self.buffer_id % self.audioBufferSize + 1)] = d
        self.audioBuffer = self.audioBuffer[1:]
        self.audioBuffer.append(d)

        buffer = np.hstack(self.audioBuffer)

        # Compute F0
        # Get power spectrum of buffer with hanning window
        if self.hann.size == buffer.size:
            # f = np.abs(np.fft.fft(buffer * self.hann)) ** 2
            f = np.abs(np.fft.fft(buffer))[:buffer.size//2]

            # Get associated frequencies
            freq = np.fft.fftfreq(buffer.size, 1/self.audioSamplingRate)[:buffer.size//2]
            #
            # self.f0_item.moveToThread(self.thread())
            if f.max() > 5:
                self.f0_item.setText("F0: {:.1f} Hz".format(freq[np.argmax(f)]))

            else:
                self.f0_item.setText("F0: xxx Hz")

    def startCamera(self):
        self.initCamera()
        self.initAudio()

        self.F0_timer.start(1)

        self.cam.getBufferSize()
        self.cam.startGrab()

        first_image = True

        while True:
            busy, status = self.cam.getStatus()

            if status == XsCamera.XS_STATUS.XSST_IDLE or status == XsCamera.XS_STATUS.XSST_REC_DONE:
                print("Status : ", status)
                # recorded_frames = self.cam.stopGrab()

                self.T = Timer(1, self.stopAudio)
                self.T.start()
                # self.stopAudio()
                self.cam.updateTriggerPosition(verbose=True)

                # print('Frames Recorded : ' + str(recorded_frames))
                break

            else:
                im = self.cam.live()

                if first_image:
                    self.im.setImage(im.transpose((1, 0, 2)))
                    first_image = False

                else:
                    self.setImage(im.transpose((1, 0, 2)))
                # self.f0_item.setText("{}".format(im.max()))
                # time.sleep(.1)
                self.app.processEvents()

    def getFrameStart(self):
        v = self.start_slider.value()

        if v >= self.end_slider.value():
            self.start_slider.setValue(v - 1)

        self.getFrame(0)

    def getFrameEnd(self):
        v = self.end_slider.value()

        if v <= self.start_slider.value():
            self.end_slider.setValue(v+1)

        self.getFrame(1)

    def getFrame(self, slider=0):
        self.updateRangeIndicator()

        if self.cam is not None:
            if slider == 0:
                sl = self.start_slider
            else:
                sl = self.end_slider

            self.cur_frame = int(sl.value()-1)
            im = self.cam.getMemoryFrame(self.cur_frame, by_trigger=True)
            self.setImage(im.transpose((1, 0, 2)))

    def analyze(self, verbose=True):
        start, end = self.start_slider.value(), self.end_slider.value()

        if verbose:
            print(start, end)

        if start == end:
            return

        elif end-start < 5:
            return

        elif self.analysis is not None:
            k = QMessageBox.question(self,
                                     "Delete analysis?",
                                     "We found an analysis already, do you want to delete this analysis and start over again?")

            if k == QMessageBox.No:
                return

            else:
                self.analysis = None
                self.imagingData = []

        self.progess.setEnabled(True)
        self.a = Analysis()
        self.a.show()

        ims = []

        for i, frame_index in enumerate(range(start-1, end)):
            im = self.cam.getMemoryFrame(frame_index, by_trigger=True)

            self.imagingData.append(im)

            im_crop = self.crop(im)
            im_crop = (im_crop - im_crop.min()) / (im_crop.max() - im_crop.min()) * 2 - 1

            self.a.segment(im_crop)

            self.progess.setValue(int(np.ceil(i / (end - start + 1e-5) * 100)))
            self.app.processEvents()

            if not self.a.isVisible():
                self.progess.setValue(0)
                self.progess.setEnabled(False)
                break

        if verbose:
            print("Total ims: ", len(ims))

        self.analysis = self.a.get()
        self.analysis['start_frame'] = start
        self.analysis['end_frame'] = end
        self.analysis['roi_pos'] = [int(i) for i in self.roi.pos()]
        self.analysis['roi_size'] = [int(i) for i in self.roi.size()]

    def save(self, verbose=True):
        if self.saveButton.text() == "Abort saving":
            self.abortSaving = True
            self.saveButton.setText("Save data.")
            self.progess.setValue(0)
            self.progess.setEnabled(False)
            QMessageBox.information(self, "Saving aborted.", "User aborted saving process. Please start again.")
            return

        self.abortSaving = False
        self.progess.setEnabled(True)

        # Get start and end range from video
        start, end = self.start_slider.value(), self.end_slider.value()

        if verbose:
            print(start, end)

        # No frames selected...
        if start == end:
            return

        # Select at least 5 frames!
        elif end - start < 5:
            return

        # No analysis found, maybe examiner wants to perform some analysis
        if self.analysis is None:
            ok = QMessageBox.question(self,
                                      "No analysis found.",
                                      "No analysis has been found. \nDo you want to save the data anyway?")

            if ok == QMessageBox.No:
                return

        # Analysis performed, but the video does not match the analysis length
        if self.analysis is not None:
            if len(self.imagingData) != (end-start+1):
                ok = QMessageBox.question(self, "Stored video",
                                        "Stored video from analysis does not match the expected length.\n"+ \
                                        "Should I keep the analysis (yes) or delete the analysis and download the video? (no)")

                if ok == QMessageBox.No:
                    self.analysis = None
                    self.imagingData = []

        from datetime import datetime
        import os
        import json
        from scipy.io.wavfile import write as wavwrite

        self.patient()

        # Saving settings
        # save_to = r"C:\openHSE"
        folder_name = self.base_folder+"\\"+self.patientInformation['folder']+"\\"+\
                      datetime.now().strftime("%Y%m%d")+"_"+self.patientInformation['last_name']

        # Create folder if it does not exist
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        # Create FN base
        now = datetime.now()
        fn_base = folder_name+"\\"+now.strftime("%Y%m%d_%H%M%S")+"_" + \
                  self.patientInformation['last_name']+"_"+self.patientInformation['first_name']

        saved = []

        # fn = QFileDialog.getSaveFileName(filter="*.mp4")[0]

        if verbose:
            print(fn_base)
            sys.stdout.flush()

        if not len(self.imagingData):
            # Get all selected frames from camera
            ims = []

            self.saveButton.setText("Abort saving")

            for i, frame_index in enumerate(range(start-1, end)):
                im = self.cam.getMemoryFrame(frame_index, by_trigger=True)
                ims.append(im)

                self.progess.setValue(int(np.ceil(i / (end - start + 1e-5) * 100)))
                self.app.processEvents()

                if self.abortSaving:
                    return

        # We already segmented everything from the camera...
        else:
            ims = self.imagingData

        if verbose:
            print("Total images: ", len(ims))

        # Save movie
        io.mimsave(fn_base+".mp4", ims)
        saved.append("Movie [mp4]")

        meta = {'Audio':
                    {
                        'Sample Rate [Hz]': self.audioSamplingRate
                    },
                'Video':
                    {
                        'Frames per second [Hz]': self.videoSamplingRate,
                        'Exposure time [us]': self.exposureTime,
                        'Start frame': self.start_slider.value(),
                        'End frame': self.end_slider.value(),
                        'Total frames recorded': self.cam.frames_to_record,
                        'Frames before trigger': self.cam.frames_before_trigger
                    },
                'Date': now.strftime("%Y-%m-%d %H:%M:%S"),
                'Patient': self.patientInformation
        }

        if self.analysis is not None:
            meta['Analysis'] = {
                'Start frame': self.analysis['start_frame'],
                'End frame': self.analysis['end_frame'],
                'ROI position': self.analysis['roi_pos'],
                'ROI size': self.analysis['roi_size']
            }

            fl.save(fn_base+".segmentation", dict(segmentation=self.analysis['segmentation']), compression=('blosc', 4))
            saved.append("Segmentation [hdf5]")

        # Write metadata to pretty printed json file
        with open(fn_base+".meta", "w+") as fp:
            json.dump(meta, fp, indent=4)
        saved.append("Metadata [json]")

        # Save last 4 s of audio
        audio = np.vstack(self.audioData)[-self.audioSamplingRate*4:]

        # Save audio as compressed hdf5 file
        fl.save(fn_base+".audio", dict(audio=audio), compression=("blosc", 5))
        saved.append("Audio [hdf5]")
        #
        # Save audio also as wav file - for the time being
        wavwrite(fn_base +".wav", self.audioSamplingRate, audio)
        saved.append("Audio [wav]")

        # io.mimsave(fn, )

        if self.save_raw:
            # fl.save(fn_base + ".raw",
            #         dict(ims=np.asarray(ims, dtype=np.uint8)),
            #         compression=("zlib", 9))
            io.mimwrite(fn_base+"_lossless.mp4",
                        ims,
                        codec='libx264rgb',
                        pixelformat='rgb24',
                        output_params=['-crf', '0',
                                       '-preset', 'ultrafast'])

            saved.append("Movie (lossless) [mp4]")
        # fl.save(fn[:-4]+"_uncompressed.h5", dict(ims=ims), compression=("blosc", 5))

        if verbose:
            print("Done with saving!")

        self.saveButton.setText("Save data.")
        self.progess.setEnabled(False)

        QMessageBox.information(self,
                                "Data saved.",
                                "Data was successfully saved here: \n{}\n\n{}".format(folder_name, "\n".join(saved)))

    def close(self):
        self.im.close()
        self.audio.close()
        super().close()


if __name__ == '__main__':
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    pix = QPixmap("openhse-01.jpg")
    splash = QSplashScreen(pix)
    splash.show()

    time.sleep(2)
    splash.close()

    w = IDTHSE(app)
    # w.show()
    w.showMaximized()
    # w.patient()
    w.updateRangeIndicator()

    sys.exit(app.exec_())
