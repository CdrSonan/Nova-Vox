def getLocale():
    locale = dict()
    settings = {}
    with open("settings.ini", 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(" ")
            settings[line[0]] = line[1]
    lang = settings["language"]
    if lang == "en":
        pass
    return locale