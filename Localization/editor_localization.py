#Copyright 2023 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from Util import SecureDict
from MiddleLayer.IniParser import readSettings

def getLanguage():
    """reads language settings and returns a dictionary with all locale-specific strings required by the editor."""

    locale = SecureDict()
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
        locale["undo_limit"] = "Undo limit:"
        #locale["hybrid"] = "hybrid"
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
        #unsaved changes warning
        locale["unsaved_changes"] = "Unsaved Changes"
        locale["unsaved_changes_msg"] = "You have unsaved changes. Do you want to save them before exiting?"

    if lang == "jp":
        #API interface
        locale["lang"] = "jp"
        #general
        locale["render_process_name"] = "Nova-Voxレンダリング方法"
        locale["all_files"] = "全てのファイル"
        #error messages and warnings
        locale["error"] = "エラー"
        locale["dataDir_err"] = "妥当のデータ登録簿がありません。"
        locale["devkit_inst_err"] = "Devkitがインストールしませんでした。"
        locale["script_err"] = "スクリプトエラー"
        locale["latency_warn_1"] = "選びオーディオデバイスは"
        locale["latency_warn_2"] = "と"
        locale["latency_warn_3"] = " 秒間の遅延の働きを図ります。下遅延を選んでければ、オーディオストリームは不安定状態になれますかアーティファクトが起きられます。"
        #builtin parameters
        locale["steadiness"] = "剛健"
        locale["breathiness"] = "ブレス"
        locale["ai_balance"] = "AIバランス"
        locale["loop_overlap"] = "ループのオバーラップ"
        locale["loop_offset"] = "ループの相殺"
        locale["vibrato_speed"] = "ヴィブラート速度"
        locale["vibrato_strength"] = "ヴィブラート強度"
        #node categories
        locale["n_math"] = "数理"
        #main UI
        locale["tempo"] = "テンポ"
        locale["quant_off"] = "Q:オフ"
        #singer settings panel
        locale["main_voice"] = "メイン声:"
        locale["mixin_voice"] = "ミックス声:"
        locale["pause_thrh"] = "間の敷居:"
        #file side panel
        locale["save"] = "保存（S)"
        locale["load"] = "開く(O)"
        locale["import"] = "取り込み(I)"
        locale["export"] = "書き出し(E)"
        locale["render"] = "レンダー"
        #file render popup
        locale["format"] = "フォーマット:"
        locale["bitdepth"] = "ビット深度:"
        locale["samplerate"] = "サンプリング周波数:"
        #singer side panel
        locale["name"] = "名:"
        locale["image"] = "画像:"
        locale["version"] = "バージョン:"
        locale["description"] = "叙事:"
        locale["license"] = "ライセンス:"
        #parameter side panel
        locale["type"] = "的:"
            #deprecated capacity and recurrency symbols omitted here
        #scripting side panel
        locale["run"] = "実行する"
        locale["SDK_open"] = "Devkitを開く"
        #settings side panel
        locale["language"] = "言語:"
        locale["accelerator"] = "アクセラレータ:"
        locale["undo_limit"] = "アンドゥの制限:"
        #locale["hybrid"] = "ハイブリッド"
        locale["TPU"] = "Tensorのコーア:"
        locale["enabled"] = "使用可能"
        locale["disabled"] = "使用無効"
        locale["lowspec_mode"] = "低仕様のモード:"
        locale["caching_mode"] = "キャッシュのモード:"
        locale["save_ram"] = "RAMを保つ"
        locale["default"] = "デフォルト"
        locale["render_speed"] = "最良のレンダリング速度"
        locale["audio_api"] = "音声のAPI:"
        locale["audio_device"] = "音声再生デバイス:"
        locale["tgt_audio_lat"] = "的の音声遅延:"
        locale["loglevel"] = "ログ記録の強度レベル:"
        locale["datadir"] = "データの所在地:"
        locale["uiscale"] = "UI尺度:"
        locale["tool_color"] = "ツールバーの色:"
        locale["acc_color"] = "アックセントの色:"
        locale["bg_color"] = "背景の色:"
        #license panel
        locale["NV_license_title"] = "Nova-Voxの投稿者そして免許"
            #omitted placeholder text
        #unsaved changes warning
        locale["unsaved_changes"] = ""
        locale["unsaved_changes_msg"] = ""
        
    return locale