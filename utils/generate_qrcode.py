#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2025-03-27 22:31:28             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

if __name__ == '__main__':

    import sys

    import matplotlib.pyplot as plt
    import qrcode

    data = sys.argv[1]
    image = qrcode.make(data)
    plt.imshow(image,  cmap='grey')
    plt.show()
