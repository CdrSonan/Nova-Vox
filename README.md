# Nova-Vox
Main repository of the Nova-Vox vocal synthesizer project.

To run the project, run the command "pip install -r requirements.txt" in an environment with access to Python 3.9.
You may need to install PyTorch from its website separately, since most of its versions are not listed on the Python Package Index.
Afterwards, execute "editor_runtime.py" or "devkit_runtime.py" for the editor and devkit respectively.

To package the project into a .exe file, run the command "pyinstaller Nova-Vox.spec" from the same environment and folder that you use to run the program.
The executable will be generated in /dist, along with its dependencies.

Afterwards, you can compile the executable into a Windows installer by compiling "installer.iss" with Inno-Setup.
