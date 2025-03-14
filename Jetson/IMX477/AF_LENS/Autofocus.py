# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import numpy as py
import os
from Focuser import Focuser

focuser = None


def focusing(val):
    # value = (val << 4) & 0x3ff0
    # data1 = (value >> 8) & 0x3f
    # data2 = value & 0xf0
    # os.system("i2cset -y 6 0x0c %d %d" % (data1,data2))
    focuser.set(Focuser.OPT_FOCUS, val)


def sobel(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_sobel = cv2.Sobel(img_gray, cv2.CV_16U, 1, 1)
    return cv2.mean(img_sobel)[0]


def laplacian(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_sobel = cv2.Laplacian(img_gray, cv2.CV_16U)
    return cv2.mean(img_sobel)[0]


# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

W: int = 3840
H: int = 2160
FPS: float = 30

# W: int = 3840
# H: int = 2160
# FPS: float = 30.0


def gstreamer_pipeline(
    sendor_id: int,
    capture_width=W,
    capture_height=H,
    display_width=W // 2,
    display_height=H // 2,
    framerate=FPS,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sendor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    max_index = 10
    max_value = 0.0
    last_value = 0.0
    dec_count = 0
    focal_distance = 10
    sendor_id: int = 0
    focus_finished = False
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    pipeline_string = gstreamer_pipeline(sendor_id=sendor_id, flip_method=0)
    print(pipeline_string)
    cap = cv2.VideoCapture(pipeline_string, cv2.CAP_GSTREAMER)
    focusing(focal_distance)
    skip_frame = 6
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)

            if skip_frame == 0:
                skip_frame = 6
                if dec_count < 6 and focal_distance < 1000:
                    # Adjust focus
                    focusing(focal_distance)
                    # Take image and calculate image clarity
                    val = laplacian(img)
                    # Find the maximum image clarity
                    if val > max_value:
                        max_index = focal_distance
                        max_value = val

                    # If the image clarity starts to decrease
                    if val < last_value:
                        dec_count += 1
                    else:
                        dec_count = 0
                    # Image clarity is reduced by six consecutive frames
                    if dec_count < 6:
                        last_value = val
                        # Increase the focal distance
                        focal_distance += 10

                elif not focus_finished:
                    # Adjust focus to the best
                    focusing(max_index)
                    focus_finished = True
            else:
                skip_frame = skip_frame - 1
            # This also acts as
            keyCode = cv2.waitKey(16) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
            elif keyCode == 10 or keyCode == 32:
                max_index = 10
                max_value = 0.0
                last_value = 0.0
                dec_count = 0
                focal_distance = 10
                focus_finished = False
            elif keyCode and keyCode != 255:
                print(f"{keyCode=}")
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


def parse_cmdline():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--i2c-bus",
        type=int,
        nargs=None,
        default=2,  # required=True,
        help="Set i2c bus, for A02 is 6, for B01 is 7 or 8, for Jetson Xavier NX it is 9 and 10.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cmdline()
    focuser = Focuser(args.i2c_bus)
    show_camera()
