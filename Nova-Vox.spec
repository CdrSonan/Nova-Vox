# -*- mode: python ; coding: utf-8 -*-

#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from kivy_deps import sdl2, glew

specpath = os.path.dirname(os.path.abspath(SPEC))

common_excludes = ["torchvision", "altgraph", "future", "pefile", "pyinstaller"]

common_imports = ["torch", "torchaudio", "soundfile"]

common_datas = [("settings.ini", "."), ("icon/*", "icon"), ("UI/kv/*", "UI/kv"), ("UI/assets/ParamList/*", "UI/assets/ParamList"), ("UI/assets/PianoRoll/*", "UI/assets/PianoRoll"), ("UI/assets/SideBar/*", "UI/assets/SideBar"), ("UI/assets/Toolbar/*", "UI/assets/Toolbar"), ("UI/assets/TopBar/*", "UI/assets/TopBar"), ("UI/assets/TrackList/*", "UI/assets/TrackList"), ("./torchaudio", "./torchaudio")]

block_cipher = None

a_editor = Analysis(['editor_runtime.py'],
                    pathex=[specpath],
                    binaries=[],
                    datas=common_datas,
                    hiddenimports=common_imports,
                    hookspath=[],
                    hooksconfig={},
                    runtime_hooks=[],
                    excludes=common_excludes,
                    win_no_prefer_redirects=False,
                    win_private_assemblies=False,
                    cipher=block_cipher,
                    noarchive=False)

a_devkit = Analysis(['devkit_runtime.py'],
                    pathex=[specpath],
                    binaries=[],
                    datas=common_datas,
                    hiddenimports=common_imports,
                    hookspath=[],
                    hooksconfig={},
                    runtime_hooks=[],
                    excludes=common_excludes,
                    win_no_prefer_redirects=False,
                    win_private_assemblies=False,
                    cipher=block_cipher,
                    noarchive=False)

MERGE( (a_editor, 'editor_runtime', 'Nova-Vox Editor'), (a_devkit, 'devkit_runtime', 'Nova-Vox Devkit') )

splash = Splash('icon/splash_new.png',
                binaries=a_editor.binaries,
                datas=a_editor.datas,
                text_pos=(200, 167),
                text_size=18,
                text_color='purple',
                text_default='loading Bootstrapper...',
                max_img_size=(667,200))

pyz_editor = PYZ(a_editor.pure, a_editor.zipped_data, cipher=block_cipher)

exe_editor = EXE(pyz_editor,
                 a_editor.scripts, 
                 splash,
                 splash.binaries,
                 [],
                 exclude_binaries=True,
                 name='Nova-Vox Editor',
                 debug=False,
                 bootloader_ignore_signals=False,
                 strip=False,
                 upx=True,
                 console=True,
                 icon = os.path.join(specpath, 'icon\\nova-vox-logo-2-color.ico'),
                 disable_windowed_traceback=False,
                 target_arch=None,
                 codesign_identity=None,
                 entitlements_file=None )

coll_editor = COLLECT(exe_editor,
                      a_editor.binaries,
                      a_editor.zipfiles,
                      a_editor.datas, 
                      *[Tree(p) for p in (sdl2.dep_bins + glew.dep_bins)],
                      strip=False,
                      upx=True,
                      upx_exclude=[],
                      name='Nova-Vox')

pyz_devkit = PYZ(a_devkit.pure, a_devkit.zipped_data, cipher=block_cipher)

exe_devkit = EXE(pyz_devkit,
                 a_devkit.scripts, 
                 [],
                 exclude_binaries=True,
                 name='Nova-Vox Devkit',
                 debug=False,
                 bootloader_ignore_signals=False,
                 strip=False,
                 upx=True,
                 console=True,
                 icon = os.path.join(specpath, 'icon\\nova-vox-logo-black.ico'),
                 disable_windowed_traceback=False,
                 target_arch=None,
                 codesign_identity=None,
                 entitlements_file=None )

coll_devkit = COLLECT(exe_devkit,
                      a_devkit.binaries,
                      a_devkit.zipfiles,
                      a_devkit.datas, 
                      strip=False,
                      upx=True,
                      upx_exclude=[],
                      name='devkit_build')

os.replace(os.path.join(specpath, 'dist\\devkit_build\\Nova-Vox Devkit.exe'), os.path.join(specpath, 'dist\\Nova-Vox\\Nova-Vox Devkit.exe'))
