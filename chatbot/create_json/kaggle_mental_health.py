import pandas as pd

import os
import json

data_path = '/home/wyundi/Server/Courses/BIA667/Project/data/Kaggle/Mental_Health_FAQ.csv'
save_path = '/home/wyundi/Server/Courses/BIA667/Project/data/Kaggle/Mental_Health_FAQ.json'
dataframe = pd.read_csv(data_path)

def write_json(dataframe, path):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'w') as f:
        for row in dataframe.iterrows():
            dic = {}
            dic['listen'] = row[1]['Questions']
            dic['reply'] = row[1]['Answers']

            jsObj = json.dumps(dic)
            f.writelines(jsObj + '\n')

write_json(dataframe, save_path)