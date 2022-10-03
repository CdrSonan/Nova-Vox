from os import getenv, mkdir, makedirs, path as osPath
from shutil import copyfile

def readSettings(path:str = None) -> dict:
    """function for reading the Nova-Vox settings file, or arbitrary other .ini settings file.
    
    Arguments:
        path: custom path to the settings file, if given. In the current implementation, this is always NONE. THe option is intended for addons using their own settings file.
        
    Returns:
        settings: dict with settings names as keys, and the corresponding setting strings as values.
        
    readSettings() is designed to be extensible in its final implementation, to allow addons to use it for easy saving and loading of their settings in both the default settings file,
    and their own custom files. Currently, this functionality is not fully implemented."""


    #if path == None:
    #    path = osPath.join(getenv("APPDATA"), "Nova-Vox", "settings.ini")
    settings = {}
    #if osPath.exists(osPath.split(path)[0]) == False:
    #    makedirs(osPath.split(path)[0])
    #if osPath.isfile(path) == False:
    #    copyfile("settings.ini", path)
    path = "settings.ini"
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("[")or line.startswith(";"):
                continue
            line = line.split("=")
            settings[line[0].strip()] = line[1].strip()
    return settings
    
def writeSettings(path, lang, accel, tcores, lowSpec, caching, audioApi, audioDevice, loglevel, dataDir):
    """function for writing a set of settings to the Nova-Vox settings file. Not intended to be used by addons; use writeCustomSettings instead."""

    if path == None:
        path = osPath.join(getenv("APPDATA"), "Nova-Vox", "settings.ini")
    with open(path, 'r+') as f:
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
