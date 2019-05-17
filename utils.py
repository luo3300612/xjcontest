import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


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


def prepare_text_feature(path, save_path):
    filenames = os.listdir(path)
    for filename in tqdm(filenames):
        txt = pd.read_table(os.path.join(path, filename), header=None)
        out = np.zeros((24,))
        for string in txt.iloc[:, 1]:
            out += get_vector(string)
        np.save(os.path.join(save_path, filename.split('.')[0] + '.npy'), out)
