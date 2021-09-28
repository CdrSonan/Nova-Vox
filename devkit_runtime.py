# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 18:54:17 2021

@author: CdrSonan
"""

print("loading...")
import devkit_UI
print("initializing...")

settings = {}
with open("settings.ini", 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split(" ")
        settings[line[0]] = line[1]
import logging
if settings["loglevel"] == "debug":
    loglevel = logging.DEBUG
elif settings["loglevel"] == "info":
    loglevel = logging.INFO
elif settings["loglevel"] == "warning":
    loglevel = logging.WARNING
elif settings["loglevel"] == "error":
    loglevel = logging.ERROR
elif settings["loglevel"] == "critical":
    loglevel = logging.CRITICAL
else:
    print("could not read loglevel setting. Loglevel has been set to \"info\" by default.")
    loglevel = logging.INFO

logging.basicConfig(format='%(asctime)s:%(process)s:%(levelname)s:%(message)s', filename='devkit.log', encoding='utf-8', level=loglevel)
logging.info("logging service started")

rootUi = devkit_UI.RootUi()
print("initialization complete")
rootUi.mainloop()