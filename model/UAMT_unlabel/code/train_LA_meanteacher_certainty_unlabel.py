import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import GaussianBlur

from networks.vnet import VNet
from networks.TG_vnet import TGVNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.honeycomb import HoneyComb, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2024_Honeycomb_Seg/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='UAMT_unlabel', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (128,128,96)


# 根据当前的训练epoch计算当前的一致性损失权重
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

# 更新指数移动平均模型（EMA model）的参数，使其逐渐跟随原模型（model）的参数变化。
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def worker_init_fn(worker_id):
    random.seed(args.seed+worker_id)

def compute_edge_labels(label_batch, output_dir='output_slices1'):
    # Move inputs to GPU if not already
    if not label_batch.is_cuda:
        label_batch = label_batch.cuda()

    label_batch = label_batch.unsqueeze(1)  # (B, C=1, H, W, D)
    label_batch = label_batch.float()  # 转换为浮点数类型

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the Sobel operator kernels for 3D
    sobel_x = torch.FloatTensor([[[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]]).cuda()
    sobel_y = torch.FloatTensor([[[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]]).cuda()
    sobel_z = torch.FloatTensor([[[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]]).transpose(2, 3).cuda()

    # Apply Sobel operator to the volume_batch (original images)
    gradient_x = F.conv3d(label_batch, sobel_x, padding=1, stride=1)
    gradient_y = F.conv3d(label_batch, sobel_y, padding=1, stride=1)
    gradient_z = F.conv3d(label_batch, sobel_z, padding=1, stride=1)

    # Compute magnitude of the gradient
    magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2)

    # Normalize magnitude to [0, 1]
    magnitude = magnitude / torch.max(magnitude)

    # Create binary edge labels using a threshold (e.g., 0.2)
    edge_labels = torch.where(magnitude > 0.2, torch.ones_like(magnitude), torch.zeros_like(magnitude))

    # # Convert edge_labels to numpy array for saving images
    # edge_labels_np = edge_labels.squeeze(1).cpu().numpy()  # 去掉 channel 维度 (B, H, W, D)
    # label_batch_np = label_batch.squeeze(1).cpu().numpy()  # 去掉 channel 维度 (B, H, W, D)
    #
    # # Iterate over batch and slices along the third dimension (depth)
    # batch_size = label_batch_np.shape[0]  # B
    # num_slices = label_batch_np.shape[3]  # D (depth)
    #
    # # Loop through each batch and save corresponding slices
    # for b in range(batch_size):
    #     # 创建对应 batch 的输出文件夹
    #     batch_output_dir = os.path.join(output_dir, f'batch_{b}')
    #     if not os.path.exists(batch_output_dir):
    #         os.makedirs(batch_output_dir)
    #
    #     for i in range(num_slices):  # 遍历每个3D数据的每个切片 (112个)
    #         original_slice = label_batch_np[b, :, :, i]  # 提取每个 batch 的二维切片 (H, W)
    #         edge_slice = edge_labels_np[b, :, :, i]      # 提取对应的边缘切片 (H, W)
    #
    #         # 保存原始标签的切片
    #         plt.imshow(original_slice, cmap='gray')
    #         plt.axis('off')
    #         plt.savefig(os.path.join(batch_output_dir, f'slice_{i}_original.png'), bbox_inches='tight', pad_inches=0)
    #         plt.close()
    #
    #         # 保存边缘分割的切片
    #         plt.imshow(edge_slice, cmap='gray')
    #         plt.axis('off')
    #         plt.savefig(os.path.join(batch_output_dir, f'slice_{i}_edge.png'), bbox_inches='tight', pad_inches=0)
    #         plt.close()

    return edge_labels
if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # 这个函数的作用是创建并初始化一个VNet模型实例，并根据传入的参数决定是否创建一个用于指数移动平均（EMA）的模型
    # EMA模型（指数移动平均模型，Exponential Moving Average Model）是一种常用于神经网络训练过程中的技术，
    # 其目的是通过对模型参数进行指数移动平均来平滑参数更新，减少噪声的影响，从而提升模型的稳定性和泛化能力。
    # 具体来说，EMA模型的参数会在每次更新时根据一个平滑因子进行加权平均，使得近期的参数更新对EMA模型的影响更大，而远期的更新影响逐渐减小。
    def create_model(ema=False):
        # Network definition
        net = TGVNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    db_train = HoneyComb(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_test = HoneyComb(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    labeled_idxs = list(range(16))
    unlabeled_idxs = list(range(16, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    # def worker_init_fn(worker_id):
    #     random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # 根据配置选择一致性损失函数。
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        # 用于计算数据加载时间
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            # print("label_batch:",label_batch.shape)
            # print("volume_batch:", volume_batch.shape)
            # Generate edge labels
            edge_label_batch = compute_edge_labels(label_batch)

            # 生成一个与 unlabeled_volume_batch 相同形状的噪声张量，值在 [-0.2, 0.2] 之间
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            # 将噪声添加到无标签数据上，生成EMA模型的输入。
            ema_inputs = unlabeled_volume_batch + noise
            # 计算原始模型的输出
            outputs, edge_outputs = model(volume_batch)
            # 使用EMA模型计算输出，并在 with torch.no_grad() 语句块中避免计算梯度
            with torch.no_grad():
                ema_output,ema_edge_output = ema_model(ema_inputs)
            T = 8
            # 将无标签数据重复两次以进行数据增强。
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            #  计算每个重复部分的步长
            stride = volume_batch_r.shape[0] // 2
            #  创建一个用于存储预测结果的张量。
            preds = torch.zeros([stride * T, 2, 128 ,128 ,96]).cuda()
            # 进行多次推理。对每次推理添加不同的噪声
            for i in range(T//2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                #  使用EMA模型进行推理，并存储预测结果
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)],_ = ema_model(ema_inputs)
            # 对多次推理的结果进行后处理，计算预测平均值和不确定性.这些步骤结合在一起，为训练过程中引入了噪声、数据增强和不确定性估计，提高了模型的稳定性和鲁棒性
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, 128, 128, 96)
            preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
            uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)


            # 计算有标签数据的监督损失，包括交叉熵损失和Dice损失。
            ## calculate the loss
            loss_seg_ce = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            loss_seg = 0.5 * (loss_seg_ce + loss_seg_dice)
            # 边缘检测损失: 结合BCE损失和Dice损失
            edge_outputs = torch.sigmoid(edge_outputs[:labeled_bs])
            edge_bce_loss = F.binary_cross_entropy(edge_outputs, edge_label_batch[:labeled_bs])
            edge_dice_loss = losses.dice_loss(edge_outputs, edge_label_batch[:labeled_bs])
            # edge_aware_loss = edge_aware_bce_loss(edge_outputs, edge_label_batch[:labeled_bs], edge_weight=2.0)

            # 边缘检测损失
            loss_edge = 0.5 * (edge_dice_loss + edge_bce_loss)
            supervised_loss = loss_seg + loss_edge


            # 计算无标签数据的一致性损失，衡量模型预测与EMA模型预测之间的差异
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output)
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            consistency_loss = consistency_weight * consistency_dist

            loss = supervised_loss + consistency_loss


            # 计算总损失并进行反向传播和优化，同时更新EMA模型的参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            # 记录训练过程中的各种指标和信息，便于后续分析和可视化
            iter_num = iter_num + 1
            writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_edge', loss_edge, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('iteration %d : loss : %f, loss_seg: %f, loss_edge: %f, consistency_loss: %f' % (iter_num, loss.item(), loss_seg.item(), loss_edge.item(), consistency_loss.item()))

            if iter_num % 50 == 0:
                # 提取批次中第一张图像的特定切片,调整图像维度顺序，使其符合可视化要求,将单通道图像重复为三通道，便于可视化
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                # 将图像转化为网格形式，便于显示。
                grid_image = make_grid(image, 5, normalize=True)
                # 将图像写入TensorBoard
                writer.add_image('train/Image', grid_image, iter_num)

                # 这段可视化模型预测的标签。
                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                # 将标签解码为可视化格式
                image = utils.decode_seg_map_sequence(image)
                # 将解码后的标签转换为网格形式
                grid_image = make_grid(image, 5, normalize=False)
                # 将预测标签写入TensorBoard。
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                # 这段可视化真实标签。
                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                # 这段可视化模型预测的不确定性。
                image = uncertainty[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/uncertainty', grid_image, iter_num)

                # 可视化不确定性掩码。
                mask2 = (uncertainty > threshold).float()
                image = mask2[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/mask', grid_image, iter_num)
                #####
                # 可视化无标签数据的输入图像
                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # 可视化无标签数据的预测标签。
                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

                # 可视化无标签数据的真实标签。
                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            # 根据迭代次数动态调整学习率。
            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            # 定期保存模型的状态，以便后续恢复和继续训练
            if iter_num % 500 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
