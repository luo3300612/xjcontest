from utils import prepare_text_feature
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate feature of txt')
    parser.add_argument('--txt-path', type=str, default='/userhome/bigdata/train/train_visit',
                        help='txt path (default:/userhome/bigdata/train/train_visit/train)')
    parser.add_argument('--output-path', type=str, default='/userhome/bigdata/train/visit_feat',
                        help='output path (default:/userhome/bigdata/train/visit_feat)')
    args = parser.parse_args()
    input_path = args.txt_path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    prepare_text_feature(input_path,output_path)
