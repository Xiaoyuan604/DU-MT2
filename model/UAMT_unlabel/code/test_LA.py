import os
import argparse
import torch
from networks.vnet import VNet
from networks.TG_vnet import TGVNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2024_Honeycomb_Seg/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='UAMT_unlabel', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/"+FLAGS.model+"/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path +item.replace('\n', '')+"/image.h5" for item in image_list]


def test_calculate_metric(epoch_num):
    net = TGVNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(128,128,96), stride_xy=32, stride_z=24,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    iterall = [3000,3500,4000,4500,5000,5500,6000]
    for i in iterall:
        metric = test_calculate_metric(i)
        print(metric)