from os import getenv, path as osPath
def readSettings(path = None):
    if path == None:
        path = osPath.join(getenv("APPDATA"), "Nova-Vox", "settings.ini")
    settings = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("["):
                continue
            line = line.split("=")
            settings[line[0].strip()] = line[1].strip()
    return settings
def writeSettings(path, lang, accel, tcores, prerender, audioApi, audioDevice, loglevel, dataDir):
    if path == None:
        path = osPath.join(getenv("APPDATA"), "Nova-Vox", "settings.ini")
    with open(path, 'w') as f:
        f.write("[lang]" + "\n")
        f.write("language = " + lang + "\n")
        f.write("\n")
        f.write("[perf]" + "\n")
        f.write("accelerator = " + accel + "\n")
        f.write("tensorCores = " + tcores + "\n")
        f.write("intermediateOutputs = " + prerender + "\n")
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
