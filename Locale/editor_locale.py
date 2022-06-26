from MiddleLayer.IniParser import readSettings
def getLocale():
    locale = dict()
    lang = readSettings()["language"]
    if lang == "en":
        locale["render_process_name"] = "Nova-Vox rendering process"
    return locale