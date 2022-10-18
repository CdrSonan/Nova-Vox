from cgi import test


from MiddleLayer.IniParser import readSettings
global uiScale
global mainColor
global accColor
global bgColor
settings = readSettings()
uiScale = float(settings["uiScale"])
mainColor = eval(settings["mainColor"])
accColor = eval(settings["accColor"])
bgColor = eval(settings["bgColor"])
