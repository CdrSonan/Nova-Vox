# Copyright 2022 - 2024 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from torch.multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    print("loading...")
    import logging
    from os import getenv, path, makedirs
    from UI.devkit.Main import RootUi
    from MiddleLayer.IniParser import readSettings
    print("initializing...")
    loglevelstr = readSettings()["loglevel"]
    if loglevelstr == "debug":
        LOGLEVEL = logging.DEBUG
    elif loglevelstr == "info":
        LOGLEVEL = logging.INFO
    elif loglevelstr == "warning":
        LOGLEVEL = logging.WARNING
    elif loglevelstr == "error":
        LOGLEVEL = logging.ERROR
    elif loglevelstr == "critical":
        LOGLEVEL = logging.CRITICAL
    else:
        print("could not read loglevel setting. Loglevel has been set to \"info\" by default.")
        LOGLEVEL = logging.INFO

    logPath = path.join(getenv("APPDATA"), "Nova-Vox", "Logs")
    try:
        makedirs(logPath)
    except FileExistsError:
        pass

    logPath = path.join(logPath, "devkit.log")

    logging.basicConfig(format='%(asctime)s:%(process)s:%(levelname)s:%(message)s',
                        filename=logPath,
                        filemode = "w",
                        force = True,
                        level=LOGLEVEL)
    logging.info("logging service started")

    rootUi = RootUi()
    print("initialization complete")
    rootUi.mainloop()
