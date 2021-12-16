import json
import os

path = "/home/wyundi/Server/Courses/BIA667/Project/data/Kaggle/dialogs.txt"
save_path = "/home/wyundi/Server/Courses/BIA667/Project/data/Kaggle/dialogs.json"

List = []
with open(path, 'r') as f:
    List = f.readlines()

def write_json(List, path):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'w') as f:
        for i in range(len(List)):
            dic = {}
            listen_reply = List[i].split('\t')
            dic['listen'] = listen_reply[0]
            dic['reply'] = listen_reply[1][:-1]

            jsObj = json.dumps(dic)
            f.writelines(jsObj + '\n')

write_json(List[:10000], save_path)