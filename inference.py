import torch
import argparse
from main import Net
from main import Data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import multiprocessing
from tqdm import tqdm
from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('model-path', type=str,
                        help='model path')
    parser.add_argument('img_path',
                        default='/userhome/bigdata/test/test_img', type=str)
    parser.add_argument('visit_feat_path',
                        default='/userhome/bigdata/test/test_visit_feat',
                        type=str)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = Net().to(device)
    net.load_state_dict(torch.load(args.model_path))
    net.eval()
    img_files = list(Path(args.img_path).iterdir())
    inference_data = Data(img_files=img_files,
                          visit_path=args.visit_feat_path,
                          train=False,
                          transforms=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]))
    result = pd.DataFrame(columns=["id", "category"])
    with torch.no_grad():
        for sample in tqdm(inference_data):
            img = sample['img'].to(device)
            feature = sample['feature'].float().to(device)
            img_name = sample['img_name']
            output = net(img, feature)
            pred_label = torch.argmax(output, dim=1).item() + 1
            result = result.append({"id": img_name.split('.')[0], "category": pred_label})
    result.to_csv("result.csv", header=False, index=False)
