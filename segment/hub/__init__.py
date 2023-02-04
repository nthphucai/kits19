from pathlib import Path

from .hub_entries import HubEntries


def get_entries(path):
    """
    get enty point of a hub folder
    :param path: path to python module
    :return: HubEntries
    """
    path = Path(path)
    return HubEntries(path.parent, path.name.replace(".py", ""))
