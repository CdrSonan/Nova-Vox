try:
    import pyi_splash
except ImportError:
    class PseudoSplash:
        def __init__(self):
            pass
        def is_alive(self):
            return False
        def close(self):
            pass
    pyi_splash = PseudoSplash()
    pass
if pyi_splash.is_alive():
    pyi_splash.update_text("loading PyTorch libraries...")
import torch
import torch.multiprocessing as mp
if pyi_splash.is_alive():
    pyi_splash.update_text("reading settings...")
from MiddleLayer.IniParser import readSettings
settings = readSettings()
if settings["tensorCores"] == "enabled":
    tcores = True
elif settings["tensorCores"] == "disabled":
    tcores = False
else:
    print("could not read tensor core setting. Tensor cores have been disabled by default.")
    tcores = False
torch.backends.cuda.matmul.allow_tf32 = tcores
torch.backends.cudnn.allow_tf32 = tcores

if pyi_splash.is_alive():
    pyi_splash.update_text("loading Nova-Vox Backend...")
from Backend.NV_Multiprocessing.Manager import RenderManager
import sys
from os import getenv, path, makedirs
if pyi_splash.is_alive():
    pyi_splash.update_text("starting logging process...")
import logging
if settings["loglevel"] == "debug":
    loglevel = logging.DEBUG
elif settings["loglevel"] == "info":
    loglevel = logging.INFO
elif settings["loglevel"] == "warning":
    loglevel = logging.WARNING
elif settings["loglevel"] == "error":
    loglevel = logging.ERROR
elif settings["loglevel"] == "critical":
    loglevel = logging.CRITICAL
else:
    print("could not read loglevel setting. Loglevel has been set to \"info\" by default.")
    loglevel = logging.INFO

logPath = path.join(getenv("APPDATA"), "Nova-Vox", "Logs")
try:
    makedirs(logPath)
except FileExistsError:
    pass

logPath = path.join(logPath, "editor.log")

logging.basicConfig(format='%(asctime)s:%(process)s:%(levelname)s:%(message)s', filename=logPath, filemode = "w", force = True, level=loglevel)
logging.info("logging service started")

if __name__ == '__main__':
    mp.freeze_support()
    if pyi_splash.is_alive():
        pyi_splash.update_text("loading UI libraries...")
    from kivy.app import App
    from kivy.core.window import Window
    from kivy.lang import Builder
    from kivy.clock import Clock
    from kivy.config import Config
    if settings["lowSpecMode"] == "disabled":
        updateInterval = 0.25
    elif settings["lowSpecMode"] == "enabled":
        updateInterval = 2.
    else:
        print("could not read low-spec mode setting. low-spec mode has been disabled by default.")
        updateInterval = 0.25
    global manager
    global sequenceList
    global voicebankList
    global aiParamStackList
    sequenceList = []
    voicebankList = []
    aiParamStackList = []
    manager = RenderManager(sequenceList, voicebankList, aiParamStackList)
    from UI.code.editor.Main import NovaVoxUI
    class NovaVoxApp(App):
        def build(self):
            self.icon = path.join("icon", "nova-vox-logo-2-color.png")
            ui = NovaVoxUI()
            Clock.schedule_interval(ui.update, updateInterval)
            return ui
    Window.minimum_height = 500
    Window.minimum_width = 800
    Config.set('graphics', 'width', '1900')
    Config.set('graphics', 'height', '1060')
    Config.set('graphics', 'window_state', 'maximized')
    Config.write()

    Builder.load_file("UI/kv/ImageButton.kv")
    Builder.load_file("UI/kv/SingerPanel.kv")
    Builder.load_file("UI/kv/ParamPanel.kv")
    Builder.load_file("UI/kv/Note.kv")
    Builder.load_file("UI/kv/PianoRoll.kv")
    Builder.load_file("UI/kv/AdaptiveSpace.kv")
    Builder.load_file("UI/kv/SidePanels.kv")
    Builder.load_file("UI/kv/LicensePanel.kv")
    Builder.load_file("UI/kv/NovaVox.kv")

    logging.info("starting render manager")
    if pyi_splash.is_alive():
        pyi_splash.update_text("starting renderer subprocess")
    if (sys.platform.startswith('win')) == False:
        mp.set_start_method("spawn")

    from asyncio import get_event_loop
    loop = get_event_loop()

    pyi_splash.close()
    try:
        loop.run_until_complete(NovaVoxApp().async_run(async_lib='asyncio'))
    except Exception as e:
        print("An error has occured:")
        print(e)
        print("Press <Enter> to close this window.")
        input("")
    finally:
        loop.close()
    logging.info("exiting program, ending logging service")
