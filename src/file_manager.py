import os
import sys
from enum import Enum, auto

"""
Example Structure
*****************

Project_Root/
├── src/
│   └── main.py
├── resources/
│   └── ui/
│   │   └── main_window.ui
│   └── icons/
│   │   └── app.ico
│   └── data/
│       └── file.dat
└── docs/
    └── readme.md

"""


class FileType(Enum):
    ProjectRoot = auto()
    ResourcesRoot = auto()
    Source = auto()
    UI = auto()
    Icons = auto()
    Docs = auto()
    Data = auto()


def get_path_type_str(path_enum: FileType):

    if path_enum == FileType.ProjectRoot:
        return ''

    resources_folder_name = 'resources'
    if path_enum == FileType.ResourcesRoot:
        return resources_folder_name
    elif path_enum == FileType.Source:
        return 'src'
    elif path_enum == FileType.Docs:
        return 'docs'
    elif path_enum == FileType.UI:
        return os.path.join(resources_folder_name, 'ui')
    elif path_enum == FileType.Icons:
        return os.path.join(resources_folder_name, 'icons')
    elif path_enum == FileType.Data:
        return os.path.join(resources_folder_name, 'data')


def path(file_name: str, file_type: FileType = None):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
        if file_type is None:
            file_type = FileType.ProjectRoot
        if file_type is not FileType.ProjectRoot:
            base_path += os.path.sep + get_path_type_str(file_type)
    return os.path.join(base_path, file_name)
