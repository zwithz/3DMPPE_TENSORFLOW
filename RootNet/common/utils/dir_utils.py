# --------------------------------------------------------
# 3DMPPE_ROOTNET
# Copyright (c) 2019 Gyeongsik Moon
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
