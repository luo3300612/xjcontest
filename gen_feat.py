from utils import prepare_text_feature
import argparse
from multiprocessing import Process
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate feature of txt')
    parser.add_argument('--txt-path', type=str, default='/userhome/bigdata/train/train_visit/train',
                        help='txt path (default:/userhome/bigdata/train/train_visit/train)')
    parser.add_argument('--output-path', type=str, default='/userhome/bigdata/train/visit_feat',
                        help='output path (default:/userhome/bigdata/train/visit_feat)')
    args = parser.parse_args()
    input_path = args.txt_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    print(args)
    filepaths = sorted(list(Path(input_path).iterdir()))
    prepare_text_feature(filepaths, output_path)
