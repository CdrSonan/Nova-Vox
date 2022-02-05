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
import torchaudio
settings = {}
with open("settings.ini", 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split(" ")
        settings[line[0]] = line[1]
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
    pyi_splash.update_text("loading Nova-Vox Backend libraries...")
import global_consts
from Backend.VB_Components.Voicebank import LiteVoicebank as Voicebank
from Backend.DataHandler.VocalSequence import VocalSequence
from Backend.NV_Multiprocessing.Manager import RenderManager
import sys

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

logging.basicConfig(format='%(asctime)s:%(process)s:%(levelname)s:%(message)s', filename='editor.log', level=loglevel)
logging.info("logging service started")

if __name__ == '__main__':
    mp.freeze_support()
    if pyi_splash.is_alive():
        pyi_splash.update_text("loading UI libraries...")
    import tkinter.filedialog
    from kivy.app import App
    from kivy.core.window import Window
    from kivy.lang import Builder
    from kivy.clock import Clock
    class NovaVoxApp(App):
        def build(self):
            self.icon = "UI/TopBar/Logo.gif"
            ui = NovaVoxUI()
            Clock.schedule_interval(ui.update, 0.25)
            return ui
    from editor_UI import NovaVoxUI
    Window.minimum_height = 500
    Window.minimum_width = 800

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

    pyi_splash.close()
    try:
        NovaVoxApp().run()
    except:
        print("An error has occured. Press <Enter> to close this window.")
        print("")
        input("")
    logging.info("exiting program, ending logging service")
