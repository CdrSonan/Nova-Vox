# Copyright 2022, 2023 Contributors to the Nova-Vox project

# This file is part of Nova-Vox.
# Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

if __name__ == '__main__':
    try:
        import pyi_splash
        # pyi_splash is only available when the program is packaged and launched through the PyInstaller Bootstrapper.
        # to avoid issues when running the program prior to packaging, pyi_splash is replaced with the PseudoSplash dummy class, which implements all relevant methods, in that case.
    except ImportError:
        class PseudoSplash:
            """Dummy class replacing pyi_splash if the program is not run through PyInstaller"""

            def __init__(self):
                pass
            def is_alive(self):
                return False
            def close(self):
                del(self)
        pyi_splash = PseudoSplash()
    if pyi_splash.is_alive():
        pyi_splash.update_text("loading PyTorch libraries...")
    import torch
    import torch.multiprocessing as mp
    mp.freeze_support()
    if pyi_splash.is_alive():
        pyi_splash.update_text("reading settings...")
    from MiddleLayer.IniParser import readSettings
    settings = readSettings()
    if settings["tensorcores"] == "enabled":
        TCORES = True
    elif settings["tensorcores"] == "disabled":
        TCORES = False
    else:
        print("could not read tensor core setting. Tensor cores have been disabled by default.")
        TCORES = False
    torch.backends.cuda.matmul.allow_tf32 = TCORES
    torch.backends.cudnn.allow_tf32 = TCORES

    if pyi_splash.is_alive():
        pyi_splash.update_text("importing utility libraries...")
    import sys
    from os import getenv, path, makedirs, environ
    from traceback import print_exc
    if pyi_splash.is_alive():
        pyi_splash.update_text("starting logging process...")
    import logging
    if settings["loglevel"] == "debug":
        LOGLEVEL = logging.DEBUG
    elif settings["loglevel"] == "info":
        LOGLEVEL = logging.INFO
    elif settings["loglevel"] == "warning":
        LOGLEVEL = logging.WARNING
    elif settings["loglevel"] == "error":
        LOGLEVEL = logging.ERROR
    elif settings["loglevel"] == "critical":
        LOGLEVEL = logging.CRITICAL
    else:
        print("could not read loglevel setting. Loglevel has been set to \"info\" by default.")
        LOGLEVEL = logging.INFO

    logPath = path.join(getenv("APPDATA"), "Nova-Vox", "Logs")
    try:
        makedirs(logPath)
    except FileExistsError:
        pass

    logPath = path.join(logPath, "editor.log")

    logging.basicConfig(format='%(asctime)s:%(process)s:%(levelname)s:%(message)s', filename=logPath, filemode = "w", force = True, level=LOGLEVEL)
    logging.info("logging service started")

    if pyi_splash.is_alive():
        pyi_splash.update_text("loading UI libraries...")
    #environ['KIVY_TEXT'] = 'pango'
    from kivy.app import App
    from kivy.core.window import Window
    from kivy.lang import Builder
    from kivy.clock import Clock
    from kivy.config import Config
    from kivy.base import ExceptionManager
    from MiddleLayer.ErrorHandler import ErrorHandler
    if settings["lowspecmode"] == "disabled":
        UPDATEINTERVAL = 0.25
    elif settings["lowspecmode"] == "enabled":
        UPDATEINTERVAL = 2.
    else:
        print("could not read low-spec mode setting. low-spec mode has been disabled by default.")
        UPDATEINTERVAL = 0.25
    from UI.code.editor.Main import NovaVoxUI
    class NovaVoxApp(App):
        def build(self):
            self.icon = path.join("assets/icon", "nova-vox-logo-2-color.png")
            ui = NovaVoxUI()
            Clock.schedule_interval(ui.update, UPDATEINTERVAL)
            return ui
    Window.minimum_height = 500
    Window.minimum_width = 800
    from csv import reader
    uiCfg = {}
    try:
        with open(path.join(settings["datadir"], "ui.cfg"), "r") as f:
            uiCfgReader = reader(f)
            for line in uiCfgReader:
                if line == []:
                    continue
                uiCfg[line[0]] = line[1]
    except FileNotFoundError:
        uiCfg = {"windowHeight": "1060",
                 "windowWidth": "1900",
                 "windowState": "False"}
    Config.set('graphics', 'width', uiCfg["windowWidth"])
    Config.set('graphics', 'height', uiCfg["windowHeight"])
    if uiCfg["windowState"] == "True":
        windowState = "maximized"
    else:
        windowState = "visible"
    Config.set('graphics', 'window_state', windowState)
    Config.set('input', 'mouse', 'mouse,disable_multitouch')
    Config.set('kivy', 'window_icon','icon/nova-vox-logo-2-color.png' )
    Config.set('kivy', 'default_font', ['Atkinson-Hyperlegible',
                                        '../assets/UI/fonts/Atkinson-Hyperlegible-Regular-102.ttf',
                                        '../assets/UI/fonts/Atkinson-Hyperlegible-Italic-102.ttf',
                                        '../assets/UI/fonts/Atkinson-Hyperlegible-Bold-102.ttf',
                                        '../assets/UI/fonts/Atkinson-Hyperlegible-BoldItalic-102.ttf'])
    Config.write()
    Builder.load_file("assets/UI/kv/Util.kv")
    Builder.load_file("assets/UI/kv/SingerPanel.kv")
    Builder.load_file("assets/UI/kv/ParamPanel.kv")
    Builder.load_file("assets/UI/kv/Note.kv")
    Builder.load_file("assets/UI/kv/PianoRoll.kv")
    Builder.load_file("assets/UI/kv/AdaptiveSpace.kv")
    Builder.load_file("assets/UI/kv/SidePanels.kv")
    Builder.load_file("assets/UI/kv/LicensePanel.kv")
    Builder.load_file("assets/UI/kv/NodeEditor.kv")
    Builder.load_file("assets/UI/kv/NovaVox.kv")

    logging.info("starting render manager")
    if pyi_splash.is_alive():
        pyi_splash.update_text("starting renderer subprocess")
    if not sys.platform.startswith('win'):
        mp.set_start_method("spawn")

    from asyncio import get_event_loop
    loop = get_event_loop()

    ExceptionManager.add_handler(ErrorHandler())

    if pyi_splash.is_alive():
        pyi_splash.close()
    try:
        loop.run_until_complete(NovaVoxApp().async_run(async_lib='asyncio'))
    except Exception as e:
        print_exc()
        print("Irrecoverable error.")
        print("Press <Enter> to close this window.")
        input("")
    finally:
        loop.close()
    logging.info("exiting program, ending logging service")
