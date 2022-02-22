from MiddleLayer.IniParser import readSettings
def getLocale():
    locale = dict()
    lang = readSettings()["language"]
    if lang == "en":
        pass
    return locale