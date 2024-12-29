import torch
import numpy as np
import os, argparse
from MyNet import COD as Network
from utils.dataloader import test_dataset
import cv2
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/BGMR/BGMR_best_48.pth')
    parser.add_argument('--model', type=str, default='PVTv2-B2',
                        choices=['PVTv2-B1', 'PVTv2-B2', 'PVTv2-B3', 'PVTv2-B4', 'PVTv2-B5'])
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='train use gpu')
    opt = parser.parse_args()

    txt_save_path = './result/{}/'.format(opt.pth_path.split('/')[-2])
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> configs:', opt)

    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')

    cudnn.benchmark = True
    model = Network(channel=64).cuda()

    # TODO: remove FC layers from snapshots
    model.load_state_dict(torch.load(opt.pth_path), strict=False)
    model.eval()

    for _data_name in ['CAMO', 'COD10K', 'NC4K']:
        map_save_path = txt_save_path + "{}/".format(_data_name)
        os.makedirs(map_save_path, exist_ok=True)

        # data_path = './Dataset/TestDataset/{}'.format(_data_name)
        data_path = 'D:/BGMRNet_TVCJ/Dataset/TestDataset/{}'.format(_data_name)
        'D:/BGMRNet_TVCJ/Dataset/TestDataset'

        image_root = '{}/Imgs/'.format(data_path)
        gt_root = '{}/GT/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        model.eval()

        for i in range(test_loader.size):
            _, image, gt, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            glo, cam_out_2, cam_out_3, cam_out_4 = model(image)

            res = F.interpolate(cam_out_4, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)


            cv2.imwrite(map_save_path + name, res*255)
            print('>>> saving prediction at: {}'.format(map_save_path + name))
