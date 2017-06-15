import os


def sort_by_date(flist):
    flist_sorted = sorted(flist, key=os.path.getmtime)
    return flist_sorted


def folders_by_date(folder='.'):
    folders = sorted(filter(os.path.isdir, os.listdir(folder)),
                     key=os.path.getmtime)
    return folders


def files_by_date(folder='.'):
    files = sorted(filter(os.path.isfile, os.listdir(folder)),
                   key=os.path.getmtime)
    return files

