import argparse
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

import torch


def _create_onnx_mode(model_checkpoint):
    try:
        net = torch.load(model_checkpoint, map_location='cpu')
        net.eval()
        dummy_input = torch.randn(1, 1, 28, 28)
        torch.onnx.export(net, dummy_input, model_checkpoint + ".proto", verbose=True)

    except Exception as E:
        print("Please verify! ", str(E))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Onnx Model Creator')
    parser.add_argument('--model', '-m', type=str,
                        help='The checkpoint location which we want to create a Onnx model of')
    args = parser.parse_args()
    _create_onnx_mode(args.model)
