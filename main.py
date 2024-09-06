import argparse
import os
from libs.utils import set_seed
from model import build_model
from dataset import build_dataloader
from libs.sacp import SACP


def main():
    parser = argparse.ArgumentParser(description='SACP')
    parser.add_argument('--seed', type=int, default=2, help='seed')
    parser.add_argument('--model', type=str, default='sstn', required=True, help='model')
    parser.add_argument('--data_name', '-s', type=str, default='ip', required=True, help='dataset name.')
    parser.add_argument('--alpha', type=float, default=0.05, required=True, help="error rate")
    parser.add_argument('--base_score', type=str, default='APS', required=True, help='base_score')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = main()
    set_seed(args.seed)
    model_name = args.model
    data_name = args.data_name
    alpha = args.alpha
    base_score = args.base_score

    data_dir = './datasets'
    model_dir = f"./pretrained/{model_name}/{model_name}_{data_name}.pth"

    model = build_model(model_name, data_name)
    data_dict = build_dataloader(model_name, data_name, data_dir)
    predictor = SACP(model, model_name, model_dir, data_name, base_score, data_dict, alpha)

    print(predictor.calculate_sacp())
