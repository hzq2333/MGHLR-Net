## python test.py --tag shiq_last --pretrained ./last.ckpt -n last -o ../data/SHIQ_Dataset/test/test_C/ -s ./outputs_shiq_last -r ./
## python test.py --tag shiq_last --pretrained ./last.ckpt -o ../data/SHIQ_Dataset/test/test_C/
## python test.py --tag shiq_last --pretrained ./last.ckpt
import os, argparse

import torch
import torch.nn.functional as F

from dataset.shiq import SHIQ_Dataset
#from dataset.rd import RD_Dataset
from model import create_model
from utils import calc_RMSE
from PIL import Image

import numpy as np
from tqdm import tqdm

from skimage.metrics  import structural_similarity, peak_signal_noise_ratio, mean_squared_error,normalized_root_mse
import cv2
import argparse
import os
import csv
import datetime


def getSSIM(img1, img2):
	return structural_similarity(img1,img2,channel_axis=2,multichannel=True)
  
def getPSNR(img1, img2):
	return peak_signal_noise_ratio(img1, img2)

def getMSE(img1, img2):
	return mean_squared_error(img1,img2)

def getNRMSE(img1,img2):
    return normalized_root_mse(img1,img2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HLRNet inference', fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.add_argument('--backbone', default='b5', type=str, help='backbone name')
    parser.add_argument('--sm', default='True',action='store_true', help='whether save mask images')
    parser.add_argument('--root_shiq', default='../data/SHIQ_Dataset', type=str, help='SHIQ dataset root directory')
    parser.add_argument('--tag', required=True, type=str, help='checkpoint tag')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
    parser.add_argument('--model_name', default='MGHLR', type=str, help='model name to choose')
    parser.add_argument('--model_name', default='MGHLR', type=str, help='model name to choose')
    parser.add_argument('--pretrained', default='./pretrained/last.ckpt', type=str, help='path to pretrained model')
    #parser.add_argument("-n","--name", required=True, type=str, help="require name")  #项目名字-->f'MGHLR_{tag}'
    #parser.add_argument("-o","--original", required=True, type=str, help="require original file path")  #原图
    #parser.add_argument("-s","--contrast", required=True, type=str, help="require contrast file path")  #结果生成图 ==> output_dir
    parser.add_argument("-r","--result", default='./', type=str, help="require result file path")  #结果csv
    args = parser.parse_args()
    
    tag = args.tag
    val_root = args.root_shiq
    output_dir = f'outputs_{tag}'
    os.makedirs(output_dir, exist_ok=True)
    output_mask_dir = f'outputs_mask_{tag}'
    os.makedirs(output_mask_dir, exist_ok=True)

    model = create_model(args)
    # load checkpoint
    checkpoint = torch.load(args.pretrained, map_location=torch.device("cpu"))
    checkpoint_ = {}
    for k, v in checkpoint['state_dict'].items():
        if not k.startswith('model.'):
            continue

        k = k[6:] # remove 'model.'
        checkpoint_[k] = v

    model.load_state_dict(checkpoint_, strict=True)
    # model to gpu
    model = model.cuda()
    model.eval()
    dst = SHIQ_Dataset(val_root, 1, 0, True, None, None)
    dst.setup(stage='val')
    val_data = dst.val_dataloader()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_data)):
            highlight_img = batch['highlight'].cuda()
            mask_img = batch['mask'].cuda()
            free_img = batch['free'].cuda()

            name = os.path.join(output_dir, batch['name'][0])
            #print(name);
            pred_masks, pred_rgbs = model(highlight_img)
            pred_img = pred_rgbs[-1]
 
            # save image
            # new add 
            #name1 = os.path.join(output_dir, batch['name'][0][:-4]+'corrected.png')
            # new change name --> name1
            Image.fromarray((pred_img.detach().cpu().squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)).save(name)
            if args.sm:
                pred_mask = pred_masks[-1]
                mask_name = os.path.join(output_mask_dir, batch['name'][0][:-4]+'_mask.png')
                #mage.fromarray((pred_mask.detach().cpu().squeeze().numpy()*255).astype(np.uint8)).save(mask_name)
                
                # 将张量转换为 numpy 数组，并进行二值化处理
                pred_mask = pred_mask.detach().cpu().squeeze().numpy()
                # 使用阈值 0.5 进行二值化，大于阈值的设为 255，小于阈值的设为 0
                pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  

                # 将二值化的结果保存为图像文件
                mask_image = Image.fromarray(pred_mask)
                mask_image.save(mask_name)

    folder1 = os.path.join(args.root_shiq,'test/test_C/')
    folder2 = output_dir


    SSIMsum = 0
    PSNRsum = 0
    MSEsum = 0
    NRMSEsum =0
    count = 0

    now_time = datetime.datetime.now()
    time_str = datetime.datetime.now().strftime('%Y-%m-%d-%Hh-%mm-%Ss')
    print(time_str)

    csvName = f'MGHLR_{tag}'+"_"+time_str+"_result.csv" 
    ## csvName = args.name+"_result.csv" 
    print(csvName)
    file = args.result

 
    if not os.path.isfile(csvName):
        with open(csvName, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["image Name", "PSNR", "SSIM" ,"MSE","NRMSE"])  # 添加标题行

    for filename in os.listdir(folder1):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img1 = cv2.imread(os.path.join(folder1, filename))
            print(os.path.join(folder1, filename))
            img2 = cv2.imread(os.path.join(folder2, filename))
            print(os.path.join(folder2, filename))
            SSIMsum = SSIMsum+getSSIM(img1, img2)
            PSNRsum = PSNRsum+getPSNR(img1, img2)
            MSEsum = MSEsum+getMSE(img1, img2)
            NRMSEsum = NRMSEsum+getNRMSE(img1, img2)
            count = count+1
        
        with open(csvName, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([filename,getPSNR(img1, img2),getSSIM(img1, img2),getMSE(img1, img2),getNRMSE(img1, img2)])

    print("PSNR is %lf"%(PSNRsum/count))
    print("SSIM is %lf"%(SSIMsum/count))
    print("MSE is %lf"%(MSEsum/count))
    print("NRMSE is %lf"%(NRMSEsum/count))
    print("count is %d"%(count))

    with open(csvName, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["average",(PSNRsum/count),(SSIMsum/count),(MSEsum/count),(NRMSEsum/count)])       
