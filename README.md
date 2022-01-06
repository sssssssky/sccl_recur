## SCCL复现

论文地址：[Supporting Clustering with Contrastive Learning](https://arxiv.org/abs/2103.12953)

作者代码：[amazon-research/sccl](https://github.com/amazon-research/sccl)

#### Dependencies:

```python
python==3.6.13
pytorch==1.6.0. 
sentence-transformers==2.0.0. 
transformers==4.8.1. 
tensorboardX==2.4.1
pandas==1.1.5
sklearn==0.24.1
numpy==1.19.5
```
#### Get Started

##### 1.获取训练数据

模型的数据需要进行数据增强，可以使用[原始数据](https://github.com/rashadulrakib/short-text-clustering-enhancement/tree/master/data)运行argument文件夹下的nlp_argument.py文件进行文件增强，或直接使用本人已经[增强后数据]()，使用[text, text1, text2, label]的样本对结构作为模型的输入。

##### 2.运行代码，开始训练

```python
python main.py\
	   --train_name googlenewsTS_contextual\
       --train_data_path $path\
       --num_classes 152\
       --batch_size 400\
       --gpu gpu\
       --model distil\
       --max_len 32\
       --tempreture 0.5\
       --alpha 1\
       --lr 1e-5\
       --lr_scale 100\
       --eval_step 100\
       --max_step 3000\
       --is_use_consistancy False\
       --is_change_centers False\
       --seed 0
```





