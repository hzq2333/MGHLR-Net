# ORHLR-Net
--- ORHLR-Net: One-stage Residual Learning Network for Joint Single Image Specular Highlight Detection and Removal


文件格式

```bash
MGHLR-Net
	code #代码
		...
		README.md
	data #数据存放位置
		SHIQ_Dataset #SHIQ数据集
			test
				test A
				test B
				test C
			train
				train A
				train B
				train C
	train_logs #代码运行会自动生成，训练中的权重均会保存在这里 此处logs_shiq
		code
		version0
		version1
		...
```

### Test

Download model file from [here](https://drive.google.com/file/d/1_9HKBisV6H5TMMX3QlTY_u1iou6Qra_S/view?usp=drive_link)，and run：

```
python test.py --tag shiq_last --pretrained ./pretrained/best.ckpt
```

