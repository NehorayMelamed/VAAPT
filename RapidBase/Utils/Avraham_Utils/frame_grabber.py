from typing import List
import KYFGLib
from KYFGLib import *


class KAYA_CONFIG:
    # basically a struct holding config data
    totalFrames = 0
    buffSize = 0
    buffIndex = 0
    buffData = 0
    copyingDataFlag = -1 # ie uninitialized
    data = 0


class ImageStream:
    def __init__(self, camera_handles: List[int], buffer_handle: STREAM_HANDLE) -> None:
        self.camera_handles = camera_handles
        self.buffer_handle = buffer_handle


def stream_event_monitor(_, event):
    if (isinstance(event, KYDEVICE_EVENT_CAMERA_CONNECTION_LOST) == True):
        print("KYDEVICE_EVENT_CAMERA_CONNECTION_LOST_ID event recognized")
        print("event_id: " + str(event.deviceEvent.eventId))
        print("cam_handle: " + format(event.camHandle.get(), '02x'))
        print("device_link: " + str(event.iDeviceLink))
        print("camera_link: " + str(event.iCameraLink))
    elif (isinstance(event, KYFGLib.KYDEVICE_EVENT_CAMERA_START_REQUEST) == True):
        print("KYDEVICE_EVENT_CAMERA_START_REQUEST event recognized")
    else:
        print("Unknown event recognized")


def initialize_libs() -> None:
    KYFGLib_Initialize(KYFGLib_InitParameters())


def print_device_data() -> None:
    (KY_GetSoftwareVersion_status, soft_ver) = KY_GetSoftwareVersion()
    print(f"KYFGLib version: {soft_ver.Major}.{soft_ver.Minor}.{soft_ver.SubMinor}\n")
    if (soft_ver.Beta > 0):
        print(f"(Beta {soft_ver.Beta})")
    if (soft_ver.RC > 0):
        print(f"(RC {soft_ver.RC})")
    (_, num_cameras_found, __) = KYFG_Scan()
    print(f"Number of scan results:{num_cameras_found}\n")
    for i in range(num_cameras_found):
        (status, dev_info) = KY_DeviceInfo(i)
        if (status != FGSTATUS_OK):
            print(f"Cant retrieve device #{i} info")
        else:
            dev_name = dev_info.szDeviceDisplayName
            print(f"Device {i}: {dev_name}")


def open_grabber(stream_monitor, grabber_index: int = 0) -> FGHANDLE:
    # stream_monitor is the callback function called whenever a frame is grabbed
    (connected_fghandle,) = KYFG_Open(grabber_index)
    connection: FGHANDLE =  connected_fghandle.get()
    KYFG_CallbackRegister(connection, stream_monitor, 0) # register callback function on recieving of new frame
    KYDeviceEventCallBackRegister(connection, stream_event_monitor, 0)
    print (f"Good connection to grabber {grabber_index}, handle= {format(connection, '02x')}")
    return connection


def check_status(connection_status: int) -> None:
    if connection_status == FGSTATUS_OK:
        print("Camera was connected successfully")
    else:
        print("Something went wrong while camera connecting")
        exit(-1)


def connect_to_grabber(connection: FGHANDLE) -> ImageStream:
    (_, camera_handles) = KYFG_UpdateCameraList(connection)  # cameraHandles: List
    (connection_status,) = KYFG_CameraOpen2(camera_handles[0], None)
    check_status(connection_status)  # check that camera was successfully connected
    (_, buffer_handle) = KYFG_StreamCreateAndAlloc(camera_handles[0], 16, 0)  # buffer_handle: STREAM_HANDLE
    return ImageStream(camera_handles, buffer_handle)


def start_grabber(image_stream: ImageStream, num_frames=5) -> None:
    KYFG_CameraStart(image_stream.camera_handles[0], image_stream.buffer_handle, num_frames)
    print("Started Grabber")


def stop_grabber(image_stream: ImageStream) -> None:
    print('\r', end='')
    sys.stdout.flush()
    KYFG_CameraStop(image_stream.camera_handles[0])
    print("Closed Grabber\n")

def set_exposure_time(new_exposure_time: float, image_stream: ImageStream, restart=False):
    if restart:
        stop_grabber(image_stream)
    KYFG_SetCameraValue(image_stream.camera_handles[0], "ExposureTime", new_exposure_time)
    if restart:
        time.sleep(1)
        start_grabber(image_stream)
