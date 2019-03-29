import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d')
parser.add_argument('--gpu', '-g')
parser.add_argument('--ratio', '-r')
parser.add_argument('--rho', '-rh')
args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    os.system("python3 prepareData.py --ratio %s --dataset %s" % (args.ratio, args.dataset))
    os.system("CUDA_VISIBLE_DEVICES=%s python3 train.py --dataset %s --rho %s" % (args.gpu, args.dataset, args.rho))
    os.system("python3 auc.py")
