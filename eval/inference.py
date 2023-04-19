import argparse
import multiprocessing
import os
import yaml

import pandas as pd
import torch
import sys
sys.path.append('/opt/ml/workspace/script')

from util.dataset import TestDataset

class CustomNamespace(argparse.Namespace):
    def __init__(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, CustomNamespace(value))
            else:
                setattr(self, key, value)
                
    def flatten_namespace(self, prefix=''):
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, CustomNamespace):
                config_dict.update(value.flatten_namespace(prefix=f"{prefix}{key}."))
            else:
                config_dict[f"{prefix}{key}"] = value
        return config_dict
    
    
def load_model(model_path, device):
    model_path = os.path.join(model_path, 'best.pth')
    model = torch.load(model_path, map_location=device)
    return model


@torch.no_grad()
def inference(data_path, model_path, output_path, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_path, device).to(device)
    
    model.eval()

    img_root = os.path.join(data_path, 'images')
    info_path = os.path.join(data_path, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            
            mask_outs, gender_outs, age_outs = torch.split(model(images), [3, 2, 3], dim=1)
            pred = torch.argmax(mask_outs, dim=-1) * 6 + torch.argmax(gender_outs, dim=-1) * 3 + torch.argmax(age_outs, dim=-1)
            
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_path, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()

    # Yaml 파일 로드
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    args = CustomNamespace(config)

    data_path = args.data_path
    model_path = os.path.join(args.checkpoints_path, args.checkpoint_name)
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    inference(data_path, model_path, output_path, args)
