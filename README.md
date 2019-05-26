# xjcontest

## 预处理
图片：
* 降噪
* 去雾
* 直方图均衡化

visit：
* 转成7\*26\*24的张量

## 数据集
* 样本不均衡问题
* visit_feat 归一化

## 下一步
* 利用样本比例向量解决样本不均衡问题
* 对visit_feat 进行归一化

## log
* resnet18(txt) + resnet20(visit) 过拟合严重，acc最高大约0.66，在2k~4k iter 达到
* 18 + 20结构最好的参数是lr=0.01 momentum=0.9
* random flip 约可以提高0.2左右精度
* weight decay = 0.001 可以提高0.05左右精度

## 
* 把所有文件写在csv里读并不一定方便，尤其是将目录也写在里面，如果文件目录有更改的可能，就会很麻烦
* 不要所有trick直接加上，先把无trick的做好再一步步加trick