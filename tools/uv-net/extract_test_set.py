import shutil
import os

"""Extract the test assemblies of a certain split from the entire a03.15_bodyGraphs dataset"""

if __name__ == "__main__":
    with open("./checkpoint/0722_182144/test_set.txt") as f:
        lines = f.readlines()
        test_assemblies = [line.strip() for line in lines]

    for i in range(len(test_assemblies)):
        test_assemblies[i] = test_assemblies[i].split("//")[-1]

    for test_assembly in test_assemblies:
        source = f"./dataset/a03.15_bodyGraphs/{test_assembly}"
        dest = f"./dataset/test_set/{test_assembly}"
        try:
            shutil.copytree(source, dest)
        except:
            print("skipped")