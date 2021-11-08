### 代码测试
  - notes:
		yolox测试，速度与yolov5相比没有优势，主要是由于添加了解耦的head，只是精度比yolov5高，
			tiny和nano版本确实gpu占用率比较小，但是速度也没有明显优势；
				代码的话比较模块化，之后有空可以看看
  - 使用yolox-nano训练新数据，416x416size,转trt之后大概2.3ms左右，占gpu利用率7%(640也是一样), 转trt之前是8/9ms左右，gpu占用率为12%

-------

- **paper**: [https://arxiv.org/pdf/2107.08430v2.pdf](https://arxiv.org/pdf/2107.08430v2.pdf) 
- **code**: [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) 

### YOLOv3 baseline
在DarkNet53+SPP的结构基础上(YOLOv3+SPP)：

#### 调整训练策略
 - EMA
 - 余弦退火
 - IoU loss
 - IoU-aware分支
 - cls和obj使用BCE loss

#### 数据增强
 - RandomHorizontalFlip
 - ColorJitter
 - multi-scale

### Improvement
#### Decouple Head
将检测头的分类与定位解耦，分开。
解耦结构对比图如图2所示。
！[图2]()
 1. 提升收敛速度，如图3所示
![图3]()
 2. 解耦对于end-to-end YOLO是有必要的，如表1所示
![表1]()

使用两个并行的3x3的conv分支，在V100上，batchsize=1：
解耦的检测头会带来1.1ms的额外推理耗时 (11.6 ms vs 10.5 ms)

#### Strong data augmentation
 - Mosaic
 - Mixup
在训练的最后15个epoch关闭mosaic和mixup；
且由于使用的更强的数据增强，所有模型都从头开始训练；

#### Anchor-free
 1. anchor-based方法的问题
	 - anchor-based方法是domain-specific的，泛化性更差
	 - anchor机制会增加检测头的计算量
 2. 添加anchor-free
	 - 将每个location的预测从3减少到1，直接预测四个值（左上角-宽高）
	 - 与FCOS一样，选择目标中心的一个范围的区域为正样本
	 - 按照FPN尺度分类样本

这些改进减少了参数和GFLOPs，提高了检测器的速度，且拥有更高的精度，如表2所示
![表2]()

#### Multi positive
为了和YOLOv3保持标签分配的规则一致，上述的anchor-free改进，每个目标仅仅选择了一个正样本；

和FCOS的center sample一样，选取目标中心点的一个范围作为正样本，3x3的范围，实现了更好的精度，如表2


#### SimOTA
**[OTA](https://arxiv.org/abs/2103.14259)**
 1. loss/quality aware
 2. center prior
 3. 为每个GT选择动态数量的anchors(动态的top k)
 4. global view

OTA虽然效果很好，但是使用Sinkhorn-Knopp算法会带来25%额外的训练时间，所以作者提出SimOTA，SimOTA就是OTA的简化版，动态的选择top k策略，得到一个近似解；

SimOTA不仅减少了训练时间，而且避免了OTA中Sinkhorn-Knopp算法额外的超参数；
结果如图2所示

#### End-to-End YOLO
按照[PSS](https://arxiv.org/pdf/2101.11782.pdf)的做法，添加了两个额外的卷积层，one-label-one 标签分配，stop gradient；
由于降低了模型的精度，所以End-to-End YOLO是作为一个可选择的选项；

### Other Backbones
#### YOLOv5
采用了YOLOv5的backbone，激活函数，和neck，相同的模型缩放规则，得到了YOLOX-S，YOLOX-M，YOLOX-L，YOLOX-X模型
![表3]()

#### Tiny/Nano 检测器
YOLOX的模型size更小精度更高
![表4]()

#### Model size和data augmentation
 - 较小的模型使用更弱的数据增强
 - 较大的模型使用更强的数据增强

![表5]()

### SOTA比较
![表6]()



