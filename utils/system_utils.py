from errno import EEXIST
from os import makedirs, path


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    # in python > 3.2, equal to makedirs(folder_path, exist_ok=true)
    try:
        makedirs(folder_path)
    except OSError as exc: # Python > 2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise
