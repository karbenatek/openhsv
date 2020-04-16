import XsCamera
import numpy as np
import sys
import ctypes

@ctypes.WINFUNCTYPE(None, XsCamera.XS_FRAME, XsCamera.XS_ERROR, XsCamera.XSULONG32)
def callb(data, err, flag):
    print(data, err, flag)

class Camera:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.is_color = 0

        # Get ROI size etc
        self.width = None
        self.height = None
        self.nAddLo = None
        self.nAddHi = None
        self.frame_size = 0

        self.frames_to_record = 4000
        self.frames_before_trigger = 4000

        self.triggerFrameIndex = 0

        self.live_frame = None
        self.live_buffer = None

        self.cam, self.CamSerial, self.CamModel = self.openCamera(verbose=verbose)
        self.configCam(verbose=verbose)

    def openCamera(self, verbose=True):
        enum_list = list(XsCamera.XsEnumCameras(XsCamera.XS_ENUM_FLT.XS_EF_GE_NO | XsCamera.XS_ENUM_FLT.XS_EF_PCI_X))

        if len(enum_list) == 0:
            print("No cameras found.", file=sys.stderr)
            sys.exit(1)

        # Use first camera in list (unlikely that there are more...)
        enum_item = enum_list[0]

        # Open Camera, get info and model
        cam = XsCamera.XsOpenCamera(enum_item.nCameraId)
        CamSerial = XsCamera.XsGetCameraInfo(cam, XsCamera.XS_INFO.XSI_SERIAL)
        CamModel = XsCamera.XsGetCameraInfo(cam, XsCamera.XS_INFO.XSI_CAMERA_MODEL)

        if verbose:
            print("CamSerial : ", CamSerial)
            print("CamModel  : ", CamModel)

        return cam, CamSerial, CamModel

    def configCam(self, verbose=True):
        sys.stdout.flush()
        cfg = XsCamera.XsReadDefaultSettings(self.cam)  # Load default settings

        XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_PIX_GAIN, 1)  # Set to upper 8 bits, default: 0
        XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_GAIN, 1)  # Set to gain = 1, default: 0
        is_color = XsCamera.XsGetCameraInfo(self.cam, XsCamera.XS_INFO.XSI_SNS_TYPE)[0]

        self.is_color = is_color

        if is_color:
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_IMG_FORMAT, XsCamera.XS_IMG_FMT.XS_IF_BGR24)
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_PIX_DEPTH, 24)
        else:
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_IMG_FORMAT, XsCamera.XS_IMG_FMT.XS_IF_GRAY8)
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_PIX_DEPTH, 8)

        XsCamera.XsRefreshCameraSettings(self.cam, cfg)

        if verbose:
            print("Configured camera.")

    def setSettings(self, exposure, fps, rec_mode=XsCamera.XS_REC_MODE.XS_RM_CIRCULAR, sync=True, verbose=True):
        cfg = XsCamera.XsReadCameraSettings(self.cam)
        e = 0.5 # For some reason this is written in the SDK manual...

        # Get dimensions from camera chip, important for buffer size!
        width = XsCamera.XsGetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIWIDTH)
        height = XsCamera.XsGetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIHEIGHT)

        if width == 1440 and height == 1024:
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIX, (1440-1024)//2)
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIWIDTH, 1024)
            width = 1024

            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIY, 0)
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIWIDTH, 1024)
            height = 1024

        elif width == 1920 and height == 1080:
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIX, 0) #(1440 - 1024) // 2)
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIWIDTH, 1920)
            width = 1920

            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIY, (1080-400)//2)
            XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_ROIWIDTH, 400)
            height = 400

        frame_size = width * height * (3 if self.is_color else 1)

        self.width = width
        self.height = height
        self.frame_size = frame_size

        if verbose:
            print("Image size: {}x{}x{}".format(height, width, (3 if self.is_color else 1)))

        # Set Frame Rate and Exposure
        XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_PERIOD, int(1000000000 / fps + e))
        XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_EXPOSURE, int(1000 * exposure))  # Exp in nS
        XsCamera.XsSetParameter(self.cam, cfg, XsCamera.XS_PARAM.XSP_REC_MODE, rec_mode)

        # you need to set bayer mode if you want mono images
        # Already set in the ConfigCam...
        # XsCamera.XsSetParameter(cam, cfg, XsCamera.XS_PARAM.XSP_PIX_DEPTH, 24)  # 8 bit, RGB

        if sync:
            XsCamera.XsSetParameter(self.cam,
                                    cfg,
                                    XsCamera.XS_SYNCIN_CFG.XS_SIC_EXT_EDGE_LO,
                                    XsCamera.XS_SYNCIN_CFG.XS_SIC_EXT_EDGE_LO)

        # Update Camera Settings...
        XsCamera.XsRefreshCameraSettings(self.cam, cfg)

        # Create Buffer
        self.live_buffer = ctypes.create_string_buffer(frame_size)

        self.live_frame = XsCamera.XS_FRAME()
        self.live_frame.nBufSize = frame_size
        self.live_frame.pBuffer = ctypes.addressof(self.live_buffer)
        self.live_frame.nImages = 1

    def getBufferSize(self):
        nAddLo, nAddHi = XsCamera.XsGetCameraInfo(self.cam, XsCamera.XS_INFO.XSI_LIVE_BUF_SIZE)
        self.nAddLo = nAddLo
        self.nAddHi = nAddHi

    def getStatus(self):
        busy, status, _, _ = XsCamera.XsGetCameraStatus(self.cam)
        return busy, status



    def startGrab(self):
        XsCamera.XsMemoryStopGrab(self.cam)
        # Create Buffer
        self.userdata_buffer = ctypes.create_string_buffer(self.frame_size*40)

        self.userdata = XsCamera.XS_FRAME()
        self.userdata.nBufSize = self.frame_size*40
        self.userdata.pBuffer = ctypes.addressof(self.userdata_buffer)
        self.userdata.nImages = 40
        XsCamera.XsMemoryStartGrab(self.cam, self.nAddLo, self.nAddHi, self.frames_to_record, self.frames_before_trigger,
                                   callb, XsCamera.XS_CALLBACK_FLAGS.XS_CF_DONE, self.userdata)

    def stopGrab(self, verbose=True):
        recorded_frames = XsCamera.XsMemoryStopGrab(self.cam)

        if verbose:
            print("Recorded frames : ", recorded_frames)

        return recorded_frames

    def live(self):
        """
        Shows live image using XsMemoryPreview
        :return:
        """
        XsCamera.XsMemoryPreview(self.cam, self.live_frame)
        im = np.frombuffer(self.live_buffer, np.uint8).reshape(self.height,
                                                               self.width,
                                                               3 if self.is_color else 1)[..., ::-1]  # BGR to RGB
        return im

    def correctForTrigger(self, frame_index):
        """
        Correct frame_index for trigger index, otherwise indexing is not historically
        :param frame_index: int
        :return: corrected frame_index, int
        """
        i_ = np.arange(self.frames_to_record, dtype=np.int32)
        i_ = np.roll(i_, -self.triggerFrameIndex)

        return int(i_[frame_index])

    def getMemoryFrame(self, frame_index, by_trigger=True):
        frame = ctypes.create_string_buffer(self.frame_size)

        if by_trigger:
            frame_index = self.correctForTrigger(frame_index)

        XsCamera.XsMemoryReadFrame(self.cam, self.nAddLo, self.nAddHi, frame_index, frame)
        im = np.frombuffer(frame, np.uint8).reshape(self.height, self.width, 3 if self.is_color else 1)[..., ::-1]
        return im

    def getTriggerPosition(self):
        return XsCamera.XsMemoryReadTriggerPosition(self.cam)

    def updateTriggerPosition(self, verbose=False):
        _, idx, _ = self.getTriggerPosition()
        self.triggerFrameIndex = idx

        if verbose:
            print("New trigger index: ", idx)

        return idx

    def closeCamera(self):
        return XsCamera.XsCloseCamera(self.cam)
