# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:57:13 2021

@author: CdrSonan
"""

import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
# "packages": ["os"] is used as example only
build_exe_options = {"packages": ["os"], "excludes": ["tkinter"]}

# base="Win32GUI" should be used only for Windows GUI app
base = None
if sys.platform == "win32":
    base = "Win32GUI"

build_exe_options = {
    "includes": ["devkit_pipeline"],
    "packages": ["soundfile"],
    "bin_path_includes": ["C:\\Users\\admin\\anaconda3\\pkgs\\pytorch-1.8.1-py3.8_cpu_0"]
}

bdist_mac_options = {
    "bundle_name": "NovaVox",
}

bdist_dmg_options = {
    "volume_label": "NovaVox",
}

executables = [Executable("devkit_UI.py", base=base), Executable("vb_tester.py", base=base)]

setup(
    name = "NovaVox",
    version = "0.1",
    description = "NovaVox hybrid vocal synthesizer",
    options={
        "build_exe": build_exe_options,
        "bdist_mac": bdist_mac_options,
        "bdist_dmg": bdist_dmg_options,
    },
    executables = executables
)