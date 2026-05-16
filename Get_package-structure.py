# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:58:04 2026

@author: Hermann
"""

import os


def list_files_recursive(path='.', files=[]):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path)
        else:
            files.append(full_path)
    return files


dir0 = r"E:\Sources_2010\Python_programs\RefraModel\RefraModel"
os.chdir(dir0)

files = list_files_recursive()

with open("structure.txt", "w") as fo:
    for f in files:
        if "__" in f or not files[0][-2:]=="py":
            continue
        fo.write(f"\n\nFile {f}:\n")
        with open(f, "r") as fi:
            print(f"File {f}")
            text = fi.readlines()
            for it, t in enumerate(text):
                if "def " in t and "(" in t:
                    if ':' in t:
                        i = t.index(":")
                    else:
                        i = len(t)-1
                    fo.write(f"{t[:i]} (line {it+1})\n")
                elif "class " in t and "(" in t:
                    if ':' in t:
                        i = t.index(":")
                    else:
                        i = len(t)-1
                    fo.write(f"{t[:i]} (line {it+1})\n")
