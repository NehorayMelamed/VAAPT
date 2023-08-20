from frame_grabber import initialize_libs, print_device_data, FGHANDLE, open_grabber, ImageStream, start_grabber, \
    connect_to_grabber, stop_grabber
from KYFGLib import *


def callback_function(arg1, arg2):
    global a
    a+=1
    if a == 20:
        print("ALL DONE!!!")


def clean_up(cb):
    initialize_libs()
    print_device_data()
    connection: FGHANDLE = open_grabber(cb, 0)
    image_stream: ImageStream = connect_to_grabber(connection)
    time.sleep(2)
    start_grabber(image_stream, 20)
    input("Press Enter to exit: ")
    stop_grabber(image_stream)


if __name__ == '__main__':
    a = 0
    clean_up(callback_function)
