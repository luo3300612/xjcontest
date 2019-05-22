import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import date
from pathlib import Path
from random import shuffle

start_day = date(2018, 10, 1)


class OutPutUtil:
    def __init__(self, terminal, log, log_file):
        self.log = log
        self.terminal = terminal
        if log:
            logging.basicConfig(level=logging.DEBUG,
                                filename=log_file,
                                filemode='a',
                                format='%(asctime)s - %(levelname)s: %(message)s')

    def speak(self, message):
        if self.terminal:
            print(message)
        if self.log:
            logging.info(message)


def get_vector(string):
    out = np.zeros((24,))
    time_list = get_time_list(string)
    for time in time_list:
        hour_list = time.split('|')
        for hour in hour_list:
            out[int(hour)] += 1
    return out


def get_time_list(string):
    day_list = string.split(',')
    time_list = [day.split('&')[-1] for day in day_list]
    return time_list


def prepare_text_feature(filepaths, save_path):
    for filepath in tqdm(filepaths):
        save_to = Path(save_path, filepath.name.split('.')[0] + '.npy')
        out = txt2np(filepath)
        np.save(save_to, out)
        # print("save_path",save_to)
        # print("components,save_path{},filename{}".format(save_path,str(filepath).split('.')[0]))


def txt2np(filepath):
    data_np = np.zeros((7, 26, 24), dtype=int)
    # print("filepath",filepath)
    data = pd.read_table(filepath, header=None)
    for string in data.iloc[:, 1]:
        day_list = string.split(',')
        for day in day_list:
            the_day = day.split('&')[0]
            hour_list = day.split('&')[1].split('|')
            the_day = date(int(the_day[0:4]), int(the_day[4:6]), int(the_day[6:]))
            delta = (the_day - start_day).days
            for hour in hour_list:
                week_np = delta // 7
                day_np = delta % 7
                data_np[day_np, week_np, int(hour)] += 1
    return data_np


def gen_csv(img_path, visit_path, ratio=[0.8, 0.1, 0.1]):
    classes = ['001', '002', '003', '004', '005', '006', '007', '008', '009']
    img_path = Path(img_path)
    files = list(img_path.iterdir())
    class_paths = []
    for clas in classes:
        class_path = []
        for file in files:
            if clas in file.name.split('_')[-1]:
                class_path.append(file)
        class_paths.append(class_path)

    train = pd.DataFrame(columns=('img_path', 'visit_path', 'class'))
    val = pd.DataFrame(columns=('img_path', 'visit_path', 'class'))
    test = pd.DataFrame(columns=('img_path', 'visit_path', 'class'))
    for i, class_path in enumerate(class_paths):
        num = len(class_path)
        shuffle(class_path)
        print("class:{}".format(i))
        print("train")
        for j in tqdm(range(0, int(ratio[0] * num))):
            file = class_path[j]
            filename = file.name
            im_path = str(file)
            vis_path = visit_path + '/' + filename.split('.')[0] + '.npy'
            apd = [im_path, vis_path, i]
            train = train.append(apd)
            # print("append:", apd)
        print("val")
        for j in tqdm(range(int(ratio[0] * num), int((ratio[0] + ratio[1]) * num))):
            file = class_path[j]
            filename = file.name
            im_path = str(file)
            vis_path = visit_path + '/' + filename.split('.')[0] + '.npy'
            apd = [im_path, vis_path, i]
            val = val.append(apd)
            # print("append:", apd)
        print("test:")
        for j in tqdm(range(int((ratio[0] + ratio[1]) * num), num)):
            file = class_path[j]
            filename = file.name
            im_path = str(file)
            vis_path = visit_path + '/' + filename.split('.')[0] + '.npy'
            apd = [im_path, vis_path, i]
            test = test.append(apd)
            # print("append:", apd)
    # assert len(train) + len(val) + len(test) == 120000, "{}+{}+{} should be 40000".format(len(train), len(val),
    #                                                                                      len(test))
    train.to_csv('trian.csv')
    val.to_csv('val.csv')
    test.to_csv('test.csv')
