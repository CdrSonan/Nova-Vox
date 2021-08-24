# Nova-Vox
Main repository of the Nova-Vox vocal synthesizer project.

Branches will be created per feature. All finished ones are to be merged into a "testing" branch that allows team members to pull nightly builds.
After validation and, if necessary, fixes, this branch will be merged onto master and be released with the next update cycle.

To build the project, use the command
pyinstaller -D <script name>.spec
The executable will be generated in /dist, along with a lib folder.