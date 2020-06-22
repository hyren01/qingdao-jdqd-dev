#!/usr/bin/env python
# -*- coding:utf-8 -*-

from config.config import Config
import openpyxl as xl
import io
import os
import re

# 遍历指定目录，获得目录下的所有文件
def each_file(filepath):
    file_dir_list = []
    path_dir = os.listdir(filepath)
    for all_dir in path_dir:
        child = os.path.join('%s%s' % (filepath, all_dir))
        file_dir_list.append(child)
    return file_dir_list


# 读取文件内容
def read_file(filename):
    file = io.open(filename, 'r', encoding='utf-8')
    text = file.read()
    file.close()
    return text


# 增加换行符
def __indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


if __name__ == '__main__':

    xlsx_path = "C:/Users/13616/Desktop/result.xlsx"
    workbook = xl.Workbook()
    sheet = workbook.active
    headers = ["code_num", "name", "remark", "use_instructions", "example1", "example2", "example3"]
    sheet.append(headers)
    line_array = []

    txt_path = "C:/Users/13616/Desktop/Noname2.txt"
    pattern = re.compile(r'\d\.\d')
    compele_sentence = ""
    with open(txt_path, encoding='utf-8') as lines:
        for line in lines:
            line = line.replace("\n", "")
            if line == " ":
                continue
            if line == "名称" or line == "描述" or line == "明" or line == "示例":
                continue

            if pattern.match(line):
                print("\n\t\t\t\t\t\t\t\t\t" + line + "\n")
                continue

            if line == "CAMEO":
                if len(line_array) > 0:
                    sheet.append(line_array)
                line_array = []
                print("--------------------------------------------")
            elif line.endswith(" ") and not line.endswith("。 ") and not line.endswith("。) "):        # name
                line_array.append(line)
                print(line)
                pass
            elif line.startswith("CAMEO-"):     # code_num
                line_array.append(line.split("CAMEO-")[1])
                print(line)
                pass
            elif line.endswith("。 ") or line.endswith("。") or line.endswith("。) "):     # remark、use_instructions、example1
                line = line.replace("使用说", "")
                compele_sentence = compele_sentence + line.lstrip()
                line_array.append(compele_sentence)
                print(compele_sentence)
                compele_sentence = ""
            elif line.startswith("使用说"):
                line = line.replace("使用说", "")
                compele_sentence = line.lstrip()
            else:
                compele_sentence = compele_sentence + line
    workbook.save(xlsx_path)

pass
