#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2025-05-13 06:42:08             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import time

import cv2


def show_temp(image):
    # Show window for 5 secs and then close the window
    cv2.imshow('image1', image)
    cv2.waitKey(5000)


# Show window until Escape/Enter is pressed
def show_forever(image):
    while True:
        # Show image
        cv2.imshow('image2', image)
        # Key press
        res = cv2.waitKey(0)
        if res == 27:  # Escape
            break
        elif res == 13:  # Enter
            break


if __name__ == '__main__':

    image = cv2.imread('/home/jordan/Pictures/Android_Filesystem.jpeg')
    image = cv2.resize(image, (720, 480))
    show_temp(image)
    time.sleep(5)
    show_forever(image)
