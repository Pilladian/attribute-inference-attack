# Python 3.8.5


import pandas
import os


def create_csv(root):
    data = pandas.DataFrame(columns=["img_file", "age"])
    data["img_file"] = os.listdir(root)

    try:
        os.system(f'rm {root}/data.csv')
    except:
        pass

    for idx, i in enumerate(os.listdir(root)):
        l = i.split('_')
        # female
        # if l[1] == '1':
        #     data["gender"][idx] = 1
        # # male
        # elif l[1] == '0':
        #     data["gender"][idx] = 0
        data["age"][idx] = int(l[0])

    data.to_csv(f"{root}data.csv", index=False, header=True)


create_csv('train/')
create_csv('eval/')
create_csv('test/')
