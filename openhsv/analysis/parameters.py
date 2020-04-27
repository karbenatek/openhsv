from PyQt5.QtWidgets import QWidget, QGridLayout
import pyqtgraph as pg
import numpy as np
from scipy.signal import find_peaks, filtfilt, butter
from datetime import datetime

def _find_bottom(x, t=0.02):
    """Finds bottom of a decaying function
    
    :param x: input array
    :type x: numpy.ndarray
    :param t: threshold deviation from minimum, defaults to 0.02
    :type t: float, optional
    :return: index of bottom
    :rtype: int
    """
    the_min = np.min(x)
    
    for i in range(len(x)):
        if (x[i]-the_min) > t*the_min:
            continue
        else:
            break
            
    return i

def detectOpeningAndClosingEvents(signal, p_max, t=0.02):
    """Detects glottis opening and closing events relative to 
    maximum opening events.
    
    :param signal: glottal area waveform
    :type signal: numpy.ndarray
    :param p_max: maxima of glottal area waveform
    :type p_max: list of indexes
    :param t: threshold for finding signal bottom, defaults to 0.02
    :type t: float, optional
    :return: opening and closing point for each maximum
    :rtype: tuple(list(int), list(int))
    """
    opening = []
    closing = []

    for i in range(len(p_max)):
        # Search for bottom in local boundary, max. to adjacent peak
        # or to the beginning or end of the signal sequence
        lower_bound = p_max[i-1] if i else 0
        higher_bound = p_max[i+1] if i < len(p_max)-1 else len(signal)

        # Run down the signal hill until bottom is found
        starts_opening = _find_bottom(y[lower_bound:p_max[i]][::-1], t=t)
        stops_closing  = _find_bottom(y[p_max[i]:higher_bound], t=t)

        # Correct for location
        starts_opening = p_max[i]-starts_opening-1
        stops_closing  = p_max[i]+stops_closing

        opening.append(starts_opening)
        closing.append(stops_closing)

    return opening, closing

def detectMaximaMinima(s, distance=5, rel_height=.35):
    """Detect maxima and minima from a signal ``s``.
    
    :param s: signal
    :type s: numpy.ndarray
    :param distance: distance between two peaks in samples, defaults to 5
    :type distance: int, optional
    :param rel_height: minimum relative height of a found peak, defaults to .35
    :type rel_height: float, optional
    :return: tuple of maxima and minima locations
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    h_max = s.max() - s.min()
    h_min = (-s).max() - (-s).min()

    p_max = find_peaks(s, 
        distance=distance, 
        height=rel_height*h_max)[0]
    
    p_min = find_peaks(-s, 
        distance=distance, 
        height=rel_height*h_min)[0]

    return p_max, p_min

def F0fromCycles(T, verbose=False):
    r"""determine fundamental frequency (F0) based on period lengths
    
    :param T: periods
    :type T: numpy.ndarray
    :param verbose: prints F0 mean and standard deviation, defaults to False
    :type verbose: bool, optional
    :return: mean of F0 and std of F0
    :rtype: tuple(float, float)
    """
    freq_est = 1/T
    f0 = freq_est.mean()
    f0_std = freq_est.std()

    if verbose:
        print("F0: {:.2f} ± {:.2f} Hz".format(f0, f0_std))
        
    return f0, f0_std

def meanJitter():
    r"""Calculating the mean jitter in ms from signal periods

    .. math::
        \text{mean-Jitter} = \frac{\sum_{i=1}^{N-1}|T_i - T_{i-1}|}{N-1}
    
    :return: mean jitter in ms
    :rtype: float
    """
    return np.mean(np.abs(np.diff(T))) * 1000

def jitterPercent(T): 
    r"""Calculating the jitter in percent from signal periods.
    
    .. math::
        \text{Jitter[%]} = \frac{\frac{1}{N-1}\sum_{i=1}^{N-1}|T_i - T_{i-1}|}{\frac{1}{N}\sum_{i=0}^{N-1}T_i} \cdot 100.
    
    :param T: The signal periods
    :type T: numpy.ndarray or list
    :return: jitter in percent
    :rtype: float
    """
    numerator   = np.mean(np.abs(T[1:]-T[:-1]))
    denominator = np.mean(T)
    
    return numerator / denominator * 100

def shimmer():
    pass

def shimmerPercent(A, e=1e-5):
    r"""Calculating the shimmer in percent from the signal amplitude maxima.
    
    .. math::
        \text{Shimmer[%]} = \frac{\frac{20}{N-1}\sum_{i=0}^{N-2}|\log_{10} \left[ \frac{A_i}{A_{i+1}} \right]|}{\frac{20}{N}\sum_{i=0}^{N-1}|\log_{10} A_i|} \cdot 100.

    :param A: The signal amplitude maxima
    :type A: numpy.ndarray or list
    :return: shimmer in percent
    :rtype: float
    """
    numerator   = np.mean([np.abs(np.log10(A[i]/A[i+1])) for i in range(len(A)-1)])
    denominator = np.mean(np.abs(np.log10(A)))

    return numerator / (denominator+e) * 100

def periodPerturbationFactor(T):
    r"""Calculating the Period Perturbation Factor (PPF) in arbitrary units using the signal periods.

    .. math::
        \text{PPF} = \frac{1}{N-1} \sum_{i=1}^{N-1}|\frac{T_i-T_{i-1}}{T_i}| \cdot 100
    
    :param T: [description]
    :type T: [type]
    :return: PPF in percent
    :rtype: float
    """
    return np.mean(np.abs(np.diff(T)/T[1:])) * 100

class Signal:
    def __init__(self, raw_signal, dt=1/4000, verbose=True):
        r"""Inits the signal class with the raw signal, e.g. audio data or glottal area waveform.
        
        :param raw_signal: Audio signal, GAW or alike
        :type raw_signal: numpy.ndarray
        """
        self.raw_signal = raw_signal
        self.dt = dt

        # Time
        # First sample is acquired after dt,
        # in total N dt-spaced samples
        self.t = np.linspace(dt, dt*len(raw_signal), len(raw_signal))
        self.verbose = verbose

        # Filtered signal to remove noise
        self.filtered_signal = raw_signal

        # Prepare feature vectors
        # Raw peaks [maxima, minima]
        # T: Periods from maxima
        # A: Amplitude maxima
        self.raw_peaks = None
        self.T = None
        self.A = None

        # FFT of raw signal
        self.fft = None
        # Corresponding frequencies to FFT
        self.fftfreq = None
        # Peaks of the power spectrum
        self.fft_peaks = None

        # Cepstrum, FFT of FFT
        self.cepstrum = None
        self.cepstrumfreq = None

    def computeFFT(self, use_filtered_signal=True, use_hanning=True):
        signal = (self.filtered_signal if use_filtered_signal else self.raw_signal).copy()

        if use_hanning:
            signal *= np.hanning(len(signal))

        self.fft = np.fft.rfft(signal)
        self.fftfreq = np.fft.rfftfreq(len(signal), d=self.dt)

    def computeCepstrum(self):
        assert self.fft is not None, "before calculating the cepstrum, first calculate the FFT"
        self.cepstrum = np.fft.ifft(np.log(np.abs(self.fft))).real
        return self.cepstrum

    def filterSignal(self, cutoff_frequency=.1):
        b, a = butter(3, cutoff_frequency)
        self.filtered_signal = filtfilt(b, a, self.raw_signal)

    def detectCycles(self, method='peaks', peak='max', use_filtered_signal=True):
        """Detects cycles using different methods.
        
        :param method: 
            method to detect cycles, defaults to 'peaks'

            - *peaks*
                Using a peak finding algorithm on the raw signal
            - *autocorrelation*
                Detects cycles and period using autocorrelation
        :type method: str, optional
        :param use_filtered_signal: uses filtered signal, if available.
        :type use_filtered_signal: bool, optional
        """
        assert method in ('peaks', 'autocorrelation'), "selected method ({}) not available!".format(method)
        assert peak in ('max', 'min'), "selected peak should be either `min` or `max`"

        signal = self.filtered_signal if use_filtered_signal else self.raw_signal
        peak = 0 if peak == 'max' else 1

        if method == 'peaks':
            self.raw_peaks = detectMaximaMinima(signal)
            
        self.T = np.diff(self.t[self.raw_peaks[peak]])
        self.A = signal[self.raw_peaks[peak]]

    def detectPhases(self, use_filtered_signal=True):
        """Detects opening and close phase in each cycle.

        :param use_filtered_signal: Event detection on raw (False) or filtered (True) signal.
        :type use_filtered_signal: bool, optional
        """
        assert self.raw_peaks, "first you need to find local minima and maxima (--> detectCycles)"

        signal = self.filtered_signal if use_filtered_signal else self.signal
        self.opening, self.closing = detectOpeningAndClosingEvents(signal, self.raw_peaks[0])

    def getPowerSpectrum(self):
        return self.fftfreq, np.abs(self.fft)

    def getCepsturm(self):
        return self.fftfreq, self.cepstrum

class AnalysisPlatform(QWidget):
    def __init__(self, raw_signal, dt):
        super().__init__()

        self._logs = []
        self.l = QGridLayout(self)

        ####################
        ### Create Signal
        ####################
        self.signal = Signal(raw_signal=raw_signal, dt=dt)
        self.signalPlot = pg.PlotWidget()
        self.signalCurve1 = self.signalPlot.plot(self.signal.t, self.signal.raw_signal)

        ####################
        ### Filter Signal 
        ####################
        self.signal.filterSignal()
        self.signalCurve2 = self.signalPlot.plot(self.signal.t, 
            self.signal.filtered_signal, 
            pen=pg.mkPen('m'))

        ####################
        ### Detect cycles
        ####################
        self.signal.detectCycles()
        self.signalPlot.plot(self.signal.t[self.signal.raw_peaks[0]], 
            self.signal.filtered_signal[self.signal.raw_peaks[0]],
            pen=pg.mkPen('y'))

        ####################
        ### Detect opening and closing phases
        ####################
        self.signal.detectPhases()

        i = 0
        cs = [(241, 196, 15), (231, 76, 60)]

        for o, c in zip(self.signal.opening, self.signal.closing):
            i1 = pg.PlotCurveItem(self.signal.t[o:c], np.zeros_like(self.signal.t[o:c]))
            i2 = pg.PlotCurveItem(self.signal.t[o:c], self.signal.raw_signal[o:c])
            between = pg.FillBetweenItem(i1, i2, brush=cs[i % len(cs)])
            self.signalPlot.addItem(between)
            i += 1

        ####################
        ### FFT, Cepstrum
        ####################
        self.signal.computeFFT(use_hanning=True, use_filtered_signal=True)

        self.powerSpectrum = pg.PlotWidget()
        self.l.addWidget(self.powerSpectrum)
        self.powerSpectrum.plot(*self.signal.getPowerSpectrum())

        self.signal.computeCepstrum()

        self.Cepstrum = pg.PlotWidget()
        self.Cepstrum.getPlotItem().setLogMode(False, True)
        self.l.addWidget(self.Cepstrum)
        self.Cepstrum.plot(*self.signal.getCepsturm())

        #######################
        ### Compute parameters
        #######################
        print(F0fromCycles(self.signal.T))
        print(jitterPercent(self.signal.T))
        print(shimmerPercent(self.signal.A))

        self.l.addWidget(self.signalPlot)

        self.setWindowTitle("Analysis Platform")

    def _log(self, msg):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._logs.append(
            "{}: {}".format(now, msg)
        )


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QInputDialog
    app = QApplication([])

    freq, ok = QInputDialog.getInt(QWidget(), 
        "Frequency", 
        "Enter frequency to test:")

    if not ok or freq <= 0:
        freq = 100

    x = np.arange(0, 1, 1/4000)
    y = np.sin(x * 2 * np.pi * freq - np.pi/2) + np.random.randn(len(x)) / 10
    y[y<0] = 0 # GAW!

    AP = AnalysisPlatform(y, # Signal
        np.diff(x).mean()) # dt

    AP.show()

    app.exec_()