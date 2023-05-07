#Copyright 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from MiddleLayer.IniParser import readSettings

def getLanguage():
    """reads language settings and returns a dictionary with all locale-specific strings required by the editor."""

    locale = dict()
    lang = readSettings()["language"]
    if lang == "en":
        #API interface
        locale["lang"] = "en"
        #general
        locale["render_process_name"] = "Nova-Vox rendering process"
        locale["all_files"] = "all files"
        #error messages and warnings
        locale["error"] = "error"
        locale["dataDir_err"] = "no valid data directory"
        locale["devkit_inst_err"] = "Devkit not installed"
        locale["script_err"] = "script error"
        locale["latency_warn_1"] = "The selected audio device is designed to operate at a latency between "
        locale["latency_warn_2"] = " and "
        locale["latency_warn_3"] = " seconds. Setting a lower latency may make the audio stream unstable or lead to artifacting."
        #builtin parameters
        locale["steadiness"] = "steadiness"
        locale["breathiness"] = "breathiness"
        locale["ai_balance"] = "AI balance"
        locale["loop_overlap"] = "loop overlap"
        locale["loop_offset"] = "loop offset"
        locale["vibrato_speed"] = "vibrato speed"
        locale["vibrato_strength"] = "vibrato strength"
        #node categories
        locale["n_math"] = "Math"
        #main UI
        locale["tempo"] = "tempo"
        locale["quant_off"] = "Q: off"
        #singer settings panel
        locale["main_voice"] = "main voice:"
        locale["mixin_voice"] = "mix-in voice:"
        locale["pause_thrh"] = "pause threshold:"
        #file side panel
        locale["save"] = "save"
        locale["load"] = "load"
        locale["import"] = "import"
        locale["export"] = "export"
        locale["render"] = "render"
        #file render popup
        locale["format"] = "format:"
        locale["bitdepth"] = "bitdepth:"
        locale["samplerate"] = "sample rate:"
        #singer side panel
        locale["name"] = "Name:"
        locale["image"] = "Image:"
        locale["version"] = "Version:"
        locale["description"] = "Description:"
        locale["license"] = "License:"
        #parameter side panel
        locale["type"] = "Type:"
            #deprecated capacity and recurrency symbols omitted here
        #scripting side panel
        locale["run"] = "run"
        locale["SDK_open"] = "open Devkit"
        #settings side panel
        locale["language"] = "Language:"
        locale["accelerator"] = "Accelerator:"
        locale["hybrid"] = "hybrid"
        locale["TPU"] = "Tensor Cores:"
        locale["enabled"] = "enabled"
        locale["disabled"] = "disabled"
        locale["lowspec_mode"] = "Low-spec mode:"
        locale["caching_mode"] = "Caching mode:"
        locale["save_ram"] = "save RAM"
        locale["default"] = "default"
        locale["render_speed"] = "best rendering speed"
        locale["audio_api"] = "Audio API:"
        locale["audio_device"] = "Audio device:"
        locale["tgt_audio_lat"] = "Target audio latency:"
        locale["loglevel"] = "Logging level:"
        locale["datadir"] = "Data directory:"
        locale["uiscale"] = "UI scale:"
        locale["tool_color"] = "Toolbar color:"
        locale["acc_color"] = "Accent color:"
        locale["bg_color"] = "Background color:"
        #license panel
        locale["NV_license_title"] = "Nova-Vox Contributors and Licenses"
            #omitted placeholder text
        
    return locale