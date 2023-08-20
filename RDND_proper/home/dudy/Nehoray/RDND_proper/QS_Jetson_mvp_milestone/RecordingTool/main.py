import sys, traceback, cv2
from multiprocessing import Queue, Value, Lock ,Condition, Process
import os

from GUI import GUI
from config_parser import get_model
from frame_grabber import STREAM_HANDLE
from kaya_pipeline import initialize_kaya, push_frames_queue
from frame_grabber import stop_grabber
import time
import signal


def ctrl_c_signal_handler(signal, frame):
    global model
    print('You pressed Ctrl+C - or killed me with -2')
    stop_grabber(model.image_stream)
    sys.exit(0)


def forward_callback(buffer_handle: STREAM_HANDLE, userContext):  # forwards KAYA callback, and adds parameters
    global model
    try:
        push_frames_queue(buffer_handle, model)
    except KeyboardInterrupt:
        print("Forward callback caught a keyboard interrupt!")
        stop_grabber(model.image_stream)
    except Exception as e:
        stop_grabber(model.image_stream)
        print(f"Exception!\n{e}\n{traceback.format_exc()}")


def main():
    global model
    signal.signal(signal.SIGINT, ctrl_c_signal_handler)
    frame_queue = Queue()
    frame_queue_cond = Condition(Lock())
    model = get_model(frame_queue, frame_queue_cond, forward_callback)

    # model.mp_q = frame_queue
    image_stream = initialize_kaya(forward_callback, model, 0)  # also sets model image stream, important since model owns image stream before callback runs
    model.image_stream = image_stream

    gui = GUI(model)
    model.gui = gui
    gui.start_slow_mode_window()

    current_recording_session = 0
    current_saved_frame = 0
    save_frame_directory = "/home/efcom/Desktop/QS_Jetson/RecordingTool/recorded_frames"

    try:
        while True:
            # Get a frame if 1 is available, or wait for the next available frame.
            if model.test_flag:
                print("trying to acquire")
                model.mp_cond.acquire()
                print("acquiring")
                while model.mp_q.empty():
                    model.mp_cond.wait()
                next_frame = model.mp_q.get()
                model.mp_cond.release()
                # Show image in GUI
                gui.show_image(next_frame)

                # Record the image in fast mode
                if model.fast_mode:
                    print("fast mode frame saved")
                    pass
                    # current_directory = os.path.join(save_frame_directory, str(current_recording_session))
                    # # Create the recording directory if it does not exist
                    # if not os.path.exists(current_directory):
                    #     os.makedirs(current_directory)
                    #
                    # # Write current image as raw bytes
                    # current_image_name = os.path.join(save_frame_directory, f"frame_{current_saved_frame}.raw")
                    # with open(current_image_name, 'wb') as raw_image:
                    #     raw_image.write(next_frame.tobytes())
                    #
                    # # Update recording session index and current recorded frame index
                    # current_saved_frame += 1
                    # if current_saved_frame == model.fast_params.num_frames:
                    #     # Reset frame counter, advance to next session and change back to slow mode
                    #     current_recording_session += 1
                    #     current_saved_frame = 0
                    #     gui.start_slow_mode_window()
                    #     model.switch_to_slow_mode()
                    # else:
                    #     # Keep going in fast mode for recording
                    #     current_saved_frame += 1

    except KeyboardInterrupt:
        print("Caught a Keyboard interrupt")
        stop_grabber(model.image_stream)
        sys.exit(0)
    except Exception as e:
        print(f"Exception!\n{e}\n{traceback.format_exc()}")
        stop_grabber(model.image_stream)
        sys.exit(0)


if __name__ == '__main__':
    model = None
    try:
        main()
    except Exception as e:
        print(f"Exception!\n{e}\n{traceback.format_exc()}")
        stop_grabber(model.image_stream)
        sys.exit(0)

