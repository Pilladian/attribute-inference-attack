# Python 3.8.5


import pandas
import os


def create_csv(root):
    try:
        os.system(f'rm {root}/data.csv')
    except:
        pass

    data = pandas.DataFrame(columns=["img_file", "race"])
    data["img_file"] = os.listdir(root)

    for idx, i in enumerate(os.listdir(root)):
        l = i.split('_')
        data["race"][idx] = int(l[2])

    data.to_csv(f"{root}data.csv", index=False, header=True)


#create_csv('train/')
#create_csv('eval/')
create_csv('test/')
