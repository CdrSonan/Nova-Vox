#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

print("loading...")
from UI.code.devkit.Main import RootUi
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

rootUi = RootUi()
print("initialization complete")
rootUi.mainloop()
