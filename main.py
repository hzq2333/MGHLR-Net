import os

import torch
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint,EarlyStopping

from dataset.shiq import SHIQ_Dataset

from config import get_args
from model import create_model
from utils import calc_RMSE, backup_code

from tensorboardX import SummaryWriter
from torchinfo import summary

import pytorch_msssim

def train_total_loss(input, target):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 将输入数据和目标数据移动到GPU上
    input = input.to(device)
    target = target.to(device)
    
    # 计算均方误差损失
    mse = F.mse_loss(input, target)
    
    # 计算平均绝对误差损失
    #l1 = F.l1_loss(input, target)
    
    # 计算平滑L1损失
    smooth_l1 = F.smooth_l1_loss(input, target)
    
    # 计算结构相似性损失
    ssim_loss = 1-pytorch_msssim.ssim(input, target, data_range=1.0, size_average=True)
    
    # 加权求和得到总体损失
    #total_loss = alpha * mse + beta * l1 + gamma * smooth_l1 + ssim
    ##total_loss = 0.2*mse+0.4*smooth_l1+0.4*ssim_loss
    total_loss = 0.3*smooth_l1+0.7*ssim_loss

    return total_loss


class HLRNetFramework(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hparams.learning_rate = args.lr
        self.hparams.weight_decay = args.wd
        self.save_hyperparameters()
        self.model = create_model(args)

    def forward(self, x):
        return self.model(x)

    def _calculate_train_loss(self, batch, batch_idx):
        highlight_img = batch['highlight']
        mask_img = batch['mask']
        free_img = batch['free']
        
        pred_masks, pred_rgbs = self.model(highlight_img)

        ###############add######################
        
        ###############add######################

        # mask loss
        loss_masks = []
        for i, pred_mask in enumerate(pred_masks):
            loss_mask = train_total_loss(pred_mask, mask_img)
            loss_masks.append(loss_mask)
            self.log(f'train/mask_loss_{i}', loss_mask)

        # rgb loss
        loss_rgbs = []
        for i, pr in enumerate(pred_rgbs):
            loss_rgb = train_total_loss(pr, free_img)
            loss_rgbs.append(loss_rgb)
            self.log(f'train/rgb_loss_{i}', loss_rgb)

        loss = 1.5*sum(loss_masks) + sum(loss_rgbs)

        self.log('train/train_loss', loss)
        return loss

    def _calculate_val_loss_acc(self, batch):
        highlight_img = batch['highlight']
        mask_img = batch['mask']
        free_img = batch['free']

        pred_masks, pred_rgbs = self.model(highlight_img)

        # mask loss
        loss_masks = []
        for i, pred_mask in enumerate(pred_masks):
            loss_mask = train_total_loss(pred_mask, mask_img)
            loss_masks.append(loss_mask)
            self.log(f'val/mask_loss_{i}', loss_mask)

        # rgb loss
        loss_rgbs = []
        for i, pr in enumerate(pred_rgbs):
            loss_rgb = train_total_loss(pr, free_img)
            loss_rgbs.append(loss_rgb)
            self.log(f'val/rgb_loss_{i}', loss_rgb)

        loss = loss_mask + sum(loss_rgbs)

        pred_img = pred_rgbs[-1]

        # calculate rmse
        free_img_np = free_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pred_img_np = pred_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask_img_np = mask_img.squeeze(0).permute(1, 2, 0).cpu().numpy()

        diff = calc_RMSE(free_img_np, pred_img_np)

        highlight_rmse = (diff * mask_img_np).sum()
        nonhighlight_rmse = (diff * (1 - mask_img_np)).sum()
        all_rmse = diff.sum()

        eval_highlight_sum = mask_img_np.sum()
        eval_nonhighlight_sum = (1 - mask_img_np).sum()
        eval_sum =  mask_img_np.sum() + (1 - mask_img_np).sum()

        self.log('val/val_loss', loss)
        return loss, highlight_rmse, nonhighlight_rmse, all_rmse, eval_highlight_sum, eval_nonhighlight_sum, eval_sum
        
    def training_step(self, batch, batch_idx):
        train_loss = self._calculate_train_loss(batch, batch_idx)
        info = {'loss':train_loss}
        return info


    def validation_step(self, batch, batch_idx):
        '''
        val_loss, mae, sdist, smask, ndist, nmask = self._calculate_val_loss_acc(batch)
        info = {'loss':val_loss, 'mae':torch.from_numpy(mae), 'sdist':sdist, 'smask':smask, 'ndist':ndist, 'nmask':nmask}
        info['progress_bar'] = {'mae':mae}
        return info
        '''
        val_loss, highlight_rmse, nonhighlight_rmse, all_rmse, eval_highlight_sum, eval_nonhighlight_sum, eval_sum = self._calculate_val_loss_acc(batch)
        info = {'loss':val_loss, 'highlight_rmse':highlight_rmse, 'nonhighlight_rmse': nonhighlight_rmse, 'all_rmse': all_rmse, \
                'eval_highlight_sum':eval_highlight_sum, 'eval_nonhighlight_sum':eval_nonhighlight_sum, 'eval_sum':eval_sum}
        return info
        

    def validation_epoch_end(self,outputs):
        highlight_rmse = torch.stack([torch.FloatTensor([x['highlight_rmse']]) for x in outputs]).mean()
        nonhighlight_rmse = torch.stack([torch.FloatTensor([x['nonhighlight_rmse']]) for x in outputs]).mean()
        all_rmse = torch.stack([torch.FloatTensor([x['all_rmse']]) for x in outputs]).mean()
        eval_highlight_sum = torch.stack([torch.FloatTensor([x['eval_highlight_sum']]) for x in outputs]).mean()
        eval_nonhighlight_sum = torch.stack([torch.FloatTensor([x['eval_nonhighlight_sum']]) for x in outputs]).mean()
        eval_sum = torch.stack([torch.FloatTensor([x['eval_sum']]) for x in outputs]).mean()

        all_mae = all_rmse/eval_sum
        s_mae = highlight_rmse/eval_highlight_sum
        ns_mae = nonhighlight_rmse/eval_nonhighlight_sum

        ##################Add###########################
        #val_loss, highlight_rmse, nonhighlight_rmse, all_rmse, eval_highlight_sum, eval_nonhighlight_sum, eval_sum = self._calculate_val_loss_acc(batch)
        loss = all_rmse = torch.stack([torch.FloatTensor([x['loss']]) for x in outputs]).mean()
        ## print(loss)
        self.log('loss',loss,prog_bar=True)
        ################################################        

        self.log('all_mae', all_mae, prog_bar=True)
        self.log('highlight', s_mae, prog_bar=True)
        self.log('non-highlight', ns_mae, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
                                                     self.args.lr, 
                                                     epochs=self.args.epochs, 
                                                     steps_per_epoch=self.args.steps_per_epoch, 
                                                     cycle_momentum=True, 
                                                     base_momentum=0.85, 
                                                     max_momentum=0.95, 
                                                     last_epoch=-1, 
                                                     pct_start=self.args.pct_start, 
                                                     div_factor=self.args.div_factor, 
                                                     final_div_factor=self.args.final_div_factor)
        return [optimizer], [lr_scheduler]


def main(hparams):
    ##################################################################
    # dataset
    ##################################################################
    scale_size = (hparams.scale_size, hparams.scale_size)
    crop_size = (hparams.crop_size, hparams.crop_size)
    dm = SHIQ_Dataset(data_dir=hparams.root_shiq, batch_size=hparams.bs, num_workers=hparams.n_workers, return_name=True, scale_size=scale_size, crop_size=crop_size)
    dm.setup()
    
    hparams.steps_per_epoch = len(dm.val_dataloader())
    print("hparams.steps_per_epoch:"+str(hparams.steps_per_epoch))

    _save_dir, _name = os.path.split(os.getcwd())
    _save_dir = os.path.join(_save_dir,"train_logs")
    _name = f'logs_{hparams.tag}'
    
    # _save_dir, _name = os.getcwd(), f'logs_{hparams.tag}' ##原始
    ##################################################################
    # save code
    ##################################################################
    code_save_dir = os.path.join(_save_dir, _name)
    print(code_save_dir)
    backup_code(code_save_dir)
    
    ##################################################################
    # call backs
    ##################################################################



    # learning rate
    lr_monitor = LearningRateMonitor(logging_interval='step')

    ##logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=_save_dir,
        version=None,
        name=_name
    )
    # checkpoint saver
    checkpoint_callback = ModelCheckpoint(
        #monitor='all_mae',#要监控指标的名称
        monitor='loss',#要监控指标的名称 -- val_loss
        #filename='HLRNet-{epoch:02d}-{all_mae:.6f}',
        filename='HLRNet-{epoch:02d}-{loss:.6f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )

    ## 在限定epoch数内没有提升，则训练
    # early_stop_callback = EarlyStopping(
        ##monitor='all_mae',
        # monitor='loss',
        # patience=hparams.Stop_No_improve,
        # mode='min'
    # )

    model = HLRNetFramework(hparams)

    ####################################################
    ################打印出网络结构######################
    #print(summary(HLRNetFramework(hparams), (1, 3, 200, 200),col_names=["kernel_size", "output_size", "num_params", "mult_adds"])) # 1：batch_size 3:图片的通道数 200: 图片的高宽
    ####################################################
    
    trainer = Trainer(
        max_epochs=hparams.epochs,
        gpus=len(params.gpus.split(',')),
        accelerator='gpu',
        default_root_dir=hparams.save_path,
        logger=tb_logger,
        #callbacks=[early_stop_callback,checkpoint_callback, lr_monitor],
        callbacks=[checkpoint_callback, lr_monitor],
        progress_bar_refresh_rate=1,
        precision=16 if hparams.amp else 32, 
        check_val_every_n_epoch=1, #每1个epoch验证一次
        ##overfit_batches=10,
        ##resume_from_checkpoint=checkpoint_path ##继续训练 改成 trainer.fit(model, datamodule=dm,ckpt_path=checkpoint_path)
    )

    # If True, we plot the computation graph in tensorboard
    trainer.logger._log_graph = True
    if hparams.c_train != './continue_train?':
        trainer.fit(model,datamodule=dm,ckpt_path=hparams.c_train)
    else:
        trainer.fit(model,datamodule=dm)



if __name__ == '__main__':
    params = get_args()
    seed_everything(params.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpus

    main(hparams=params)
