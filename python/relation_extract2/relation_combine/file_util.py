import os
import shutil
import numpy as np


def openfile(filename, mode='a'):
    f = open(filename, mode, encoding='utf-8')
    return f


def file2lines(filename):
    """
    read file and return its lines, with '\n' excluded
    :param filename:
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [l.replace('\n', '') for l in lines]
    return lines


def save_lines(lines, filename):
    """
    save lines to a file, the lines don't have to end with '\n'
    :param lines:
    :param filename:
    :return:
    """
    lines = [l + '\n' for l in lines]
    with open(filename, 'a', encoding='utf-8') as f:
        f.writelines(lines)


def save_array(array, dir_, filename, fmt='%.3f'):
    if not os.path.exists(dir_):
        os.mkdir(dir_)
    np.savetxt(dir_ + '/' + filename, array, fmt, delimiter=' ')


def add_head(fp):
    with open('head.txt', 'r') as f:
        head_lines = f.readlines()
    with open(fp, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.writelines(head_lines)
        f.write('\n')
        f.write(content)


if __name__ == '__main__':
    pass
