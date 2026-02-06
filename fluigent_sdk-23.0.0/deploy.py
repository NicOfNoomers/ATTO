# -*- coding: utf-8 -*-
"""
"""

from distutils.core import run_setup
import os

bitmap_image_name = "fluigent_logo_python.bmp"
bitmap_path = os.path.join("res", bitmap_image_name)
bitmap_option = '--bitmap={}'.format(bitmap_path)

run_setup("setup.py", ["sdist","--format=zip"])