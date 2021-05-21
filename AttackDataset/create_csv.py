# Python 3.8.5


import pandas
import os


def create_csv(root):
    try:
        os.system(f'rm {root}/data.csv')
    except:
        pass

    data = pandas.DataFrame(columns=["img_file", "race"])
    classes = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    img_files = []
    races = []
    idx = 0
    for _, i in enumerate(os.listdir(root)):
        l = i.split('_')
        if classes[int(l[2])] < 342:
            img_files.append(i)
            races.append(int(l[2]))
            classes[int(l[2])] += 1
            idx += 1

    data["img_file"] = img_files
    data["race"] = races

    data.to_csv(f"{root}data.csv", index=False, header=True)


create_csv('test/')
