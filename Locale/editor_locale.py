#Copyright 2022 Contributors to the Nova-Vox project

#This file is part of Nova-Vox.
#Nova-Vox is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#Nova-Vox is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with Nova-Vox. If not, see <https://www.gnu.org/licenses/>.

from MiddleLayer.IniParser import readSettings

def getLocale():
    """reads language settings and returns a dictionary with all locale-specific strings required by the editor."""

    locale = dict()
    lang = readSettings()["language"]
    if lang == "en":
        locale["render_process_name"] = "Nova-Vox rendering process"
    locale["render_process_name"] = "Nova-Vox rendering process"
    return locale