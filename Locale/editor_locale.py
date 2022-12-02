from MiddleLayer.IniParser import readSettings

def getLocale():
    """reads language settings and returns a dictionary with all locale-specific strings required by the editor."""

    locale = dict()
    lang = readSettings()["language"]
    if lang == "en":
        locale["render_process_name"] = "Nova-Vox rendering process"
    return locale