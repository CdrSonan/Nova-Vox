from os import getenv, mkdir, path as osPath
from shutil import copyfile

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
    if osPath.isfile(path) == False:
        copyfile("settings.ini", path)
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("[")or line.startswith(";"):
                continue
            line = line.split("=")
            settings[line[0].strip()] = line[1].strip()
    return settings
    
def writeSettings(path, lang, accel, tcores, lowSpec, caching, audioApi, audioDevice, loglevel, dataDir):
    """function for writing a set of settings to the Nova-Vox settings file. Will be replaced by a more general function later."""

    if path == None:
        path = osPath.join(getenv("APPDATA"), "Nova-Vox", "settings.ini")
    with open(path, 'w') as f:
        f.write("[lang]" + "\n")
        f.write("language = " + lang + "\n")
        f.write("\n")
        f.write("[perf]" + "\n")
        f.write("accelerator = " + accel + "\n")
        f.write("tensorCores = " + tcores + "\n")
        f.write("lowSpecMode = " + lowSpec + "\n")
        f.write("cachingMode = " + caching + "\n")
        f.write("\n")
        f.write("[audio]" + "\n")
        f.write("audioApi = " + audioApi + "\n")
        f.write("audioDevice = " + audioDevice + "\n")
        f.write("\n")
        f.write("[log]" + "\n")
        f.write("loglevel = " + loglevel + "\n")
        f.write("\n")
        f.write("[dirs]" + "\n")
        f.write("dataDir = " + dataDir + "\n")
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

#TODO: writer for custom settings files