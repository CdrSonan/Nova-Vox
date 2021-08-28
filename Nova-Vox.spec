# -*- mode: python ; coding: utf-8 -*-

specpath = os.path.dirname(os.path.abspath(SPEC))

common_excludes = ["pillow", "torchvision", "altgraph", "future", "pefile", "pyinstaller"]

block_cipher = None

a_editor = Analysis(['vb_tester.py'],
                    pathex=[specpath],
                    binaries=[],
                    datas=[],
                    hiddenimports=[],
                    hookspath=[],
                    hooksconfig={},
                    runtime_hooks=[],
                    excludes=common_excludes,
                    win_no_prefer_redirects=False,
                    win_private_assemblies=False,
                    cipher=block_cipher,
                    noarchive=False)

a_devkit = Analysis(['Devkit_UI.py'],
                    pathex=[specpath],
                    binaries=[],
                    datas=[],
                    hiddenimports=[],
                    hookspath=[],
                    hooksconfig={},
                    runtime_hooks=[],
                    excludes=common_excludes,
                    win_no_prefer_redirects=False,
                    win_private_assemblies=False,
                    cipher=block_cipher,
                    noarchive=False)

MERGE( (a_editor, 'vb_tester', 'Nova-Vox Editor'), (a_devkit, 'Devkit_UI', 'Nova-Vox VB Devkit') )

pyz_editor = PYZ(a_editor.pure, a_editor.zipped_data, cipher=block_cipher)

exe_editor = EXE(pyz_editor,
                 a_editor.scripts, 
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
                      strip=False,
                      upx=True,
                      upx_exclude=[],
                      name='Nova-Vox')

pyz_devkit = PYZ(a_devkit.pure, a_devkit.zipped_data, cipher=block_cipher)

exe_devkit = EXE(pyz_devkit,
                 a_devkit.scripts, 
                 [],
                 exclude_binaries=True,
                 name='Nova-Vox VB Devkit',
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

os.replace(os.path.join(specpath, 'dist\\devkit_build\\Nova-Vox VB Devkit.exe'), os.path.join(specpath, 'dist\\Nova-Vox\\Nova-Vox VB Devkit.exe'))
os.replace(os.path.join(specpath, 'dist\\devkit_build\\Nova-Vox VB Devkit.exe.manifest'), os.path.join(specpath, 'dist\\Nova-Vox\\Nova-Vox VB Devkit.exe.manifest'))
os.rmdir(os.path.join(specpath, 'dist\\devkit_build'))
