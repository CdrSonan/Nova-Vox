from MiddleLayer.IniParser import readSettings
global uiScale
global toolColor
global accColor
global bgColor
settings = readSettings()
uiScale = float(settings["uiScale"])
toolColor = eval(settings["toolColor"])
accColor = eval(settings["accColor"])
bgColor = eval(settings["bgColor"])
