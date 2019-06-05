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
    parser.add_argument('--model-path',
                        type=str,
                        default='/userhome/xjcontest/result/lr0.01m0.9train2hvflipWD0.001firstcommit/model6500',
                        help='model path')
    parser.add_argument('--img_path',
                        type=str,
                        default='/userhome/bigdata/test/test_img')
    parser.add_argument('--visit_feat_path',
                        type=str,
                        default='/userhome/bigdata/test/test_visit_feat')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = Net().to(device)
    net.load_state_dict(torch.load(args.model_path))
    net.eval()
    img_files = sorted(list(Path(args.img_path).iterdir()))
    inference_data = Data(img_files=img_files,
                          visit_path=args.visit_feat_path,
                          train=False,
                          transforms=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]))
    inference_loader = DataLoader(inference_data,
                                  batch_size=128,
                                  num_workers=multiprocessing.cpu_count(),
                                  pin_memory=True)
    result = pd.DataFrame(columns=["id", "category"])
    # distribution = torch.Tensor([0.2386, 0.1884, 0.0897, 0.0340, 0.0866, 0.1377, 0.0879, 0.0654, 0.0717]).to(device)
    with torch.no_grad():
        for idx,sample in enumerate(tqdm(inference_loader)):
            img = sample['img'].to(device)
            feature = sample['feature'].float().to(device)
            output = net(img, feature)
            # output = torch.nn.functional.softmax(output,dim=1)
            # output = output / distribution
            pred_label = torch.argmax(output, dim=1) + 1

            for i in range(idx*128,(idx+1)*128):
                if i >= 10000:
                    break
                img_name = img_files[i].name
                result = result.append({"id": img_name.split('.')[0], "category": '00'+str(pred_label[i%128].item())},ignore_index=True)
    result.to_csv("result.csv", header=False, index=False,sep='\t')
