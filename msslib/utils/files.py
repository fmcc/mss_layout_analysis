"""
Functions related to file and path manipulation.
"""
import os

def listpaths(d):
    """ Create a list of all paths for files in a directory """ 
    return [os.path.join(d, f) for f in os.listdir(d)]

def format_path(directory, extension, filename, label=''):
    """ Create a path from a directory, base filename and an extension. 
        """
    return os.path.join(directory, "%s%s.%s" %(filename, label, extension))

def only_basename(path):
    """ Splits extension and directory from filepath.
        """
    basename, ext = os.path.splitext(os.path.basename(path))
    return basename

def mkdir(path):
    """ Creates a directory if it doesn't already exist. 
        """
    if not os.path.exists(path):
        os.makedirs(path)
