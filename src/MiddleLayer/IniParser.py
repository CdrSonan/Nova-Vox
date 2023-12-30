#Copyright 2022, 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from os import getenv, mkdir, makedirs, path as osPath
from shutil import copyfile
import configparser as cp

def readSettings(path:str = None) -> dict:
    """function for reading the Nova-Vox settings file, or arbitrary other .ini settings file.
    
    Arguments:
        path: custom path to the settings file, if given. In the current implementation, this is always NONE. THe option is intended for addons using their own settings file.
        
    Returns:
        settings: dict with settings names as keys, and the corresponding setting strings as values.
        
    readSettings() is designed to be extensible in its final implementation, to allow addons to use it for easy saving and loading of their settings in both the default settings file,
    and their own custom files. Currently, this functionality is not fully implemented."""


    if path == None:
        path = osPath.join(getenv("APPDATA"), "Nova-Vox", "settings.ini")
    settings = {}
    if osPath.exists(osPath.split(path)[0]) == False:
        makedirs(osPath.split(path)[0])
    if osPath.isfile(path) == False:
        try:
            copyfile("assets/settings.ini", path)
        except PermissionError:
            path = "assets/settings.ini"
    config = cp.ConfigParser()
    config.read(path)
    for i in config.sections():
        for j in config[i].keys():
            settings[j] = config[i][j]
    return settings
    
def writeSettings(path, lang, accel, tcores, lowSpec, caching, audioApi, audioDevice, audioLatency, undoLimit, loglevel, dataDir, uiScale, toolColor, accColor, bgColor):
    """function for writing a set of settings to the Nova-Vox settings file. Not intended to be used by addons; use writeCustomSettings instead."""

    if path == None:
        path = osPath.join(getenv("APPDATA"), "Nova-Vox", "settings.ini")
    config = cp.ConfigParser()
    config["lang"] = {"language": lang,}
    config["perf"] = {"accelerator": accel,
                      "tensorcores": tcores,
                      "lowspecmode": lowSpec,
                      "cachingmode": caching}
    config["audio"] = {"audioapi": audioApi,
                       "audiodevice": audioDevice,
                       "audiolatency": str(audioLatency)}
    config["undo"] = {"undolimit": str(undoLimit),}
    config["log"] = {"loglevel": loglevel,}
    config["dirs"] = {"datadir": dataDir,}
    config["ui"] = {"uiscale": str(uiScale),
                    "toolcolor": str(toolColor),
                    "acccolor": str(accColor),
                    "bgcolor": str(bgColor)}
    with open(path, 'w') as f:
        config.write(f)
    if dataDir == "None":
        return
    voicePath = osPath.join(dataDir, "Voices")
    if osPath.isdir(voicePath) == False:
        mkdir(voicePath)
    paramPath = osPath.join(dataDir, "Parameters")
    if osPath.isdir(paramPath) == False:
        mkdir(paramPath)
    addonPath = osPath.join(dataDir, "Addons")
    if osPath.isdir(addonPath) == False:
        mkdir(addonPath)

def writeCustomSettings(category:str, data:dict, path:str = None) -> None:
    """function for allowing addons to write their own settings to the Nova-Vox settings file, or an individual .ini file.
    
    Arguments:
        category: The section of the .ini file that the settings will be stored in. By convention, this should be the name or handle of the addon using this function.
        
        data: dictionary of key-value pairs representing setting names and their respective values. All values must be strings.
        
        path: optional file path for a custom settings file."""

    if path == None:
        path = osPath.join(getenv("APPDATA"), "Nova-Vox", "settings.ini")
    with open(path, 'r+') as f:
        rightCategory = False
        for line in f:
            if rightCategory:
                for i in data.keys:
                    if line.startswith(i):
                        f.write(i + " = " + data[i] + "\n")
                        del data[i]
            if line.startswith("[" + category + "]"):
                if rightCategory:
                    break
                else:
                    rightCategory = True
        else:
            f.write(line + "\n" + "[" + category + "]\n")
        for i in data.keys:
            f.write(line + "\n" + i + " = " + data[i] + "\n")
