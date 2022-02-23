# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 18:54:17 2021

@author: CdrSonan
"""

print("loading...")
import devkit_UI
from MiddleLayer.IniParser import readSettings
print("initializing...")

loglevelstr = readSettings()["loglevel"]
import logging
from os import getenv, path, makedirs
if loglevelstr == "debug":
    loglevel = logging.DEBUG
elif loglevelstr == "info":
    loglevel = logging.INFO
elif loglevelstr == "warning":
    loglevel = logging.WARNING
elif loglevelstr == "error":
    loglevel = logging.ERROR
elif loglevelstr == "critical":
    loglevel = logging.CRITICAL
else:
    print("could not read loglevel setting. Loglevel has been set to \"info\" by default.")
    loglevel = logging.INFO

logPath = path.join(getenv("APPDATA"), "Nova-Vox", "Logs")
try:
    makedirs(logPath)
except FileExistsError:
    pass

logPath = path.join(logPath, "devkit.log")

logging.basicConfig(format='%(asctime)s:%(process)s:%(levelname)s:%(message)s', filename=logPath, filemode = "w", force = True, level=loglevel)
logging.info("logging service started")

rootUi = devkit_UI.RootUi()
print("initialization complete")
rootUi.mainloop()