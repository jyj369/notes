- **paper**: [https://arxiv.org/pdf/2111.08557v2.pdf](https://arxiv.org/pdf/2111.08557v2.pdf)
- **code**: [https://github.com/wmcnally/kapao](https://github.com/wmcnally/kapao)

## Notes
- labels(I guess)
  ```vim
  0 cx, cy, w, h, (kx, ky, kc)*17
  1 cx, cy, w, h, (0, 0, 0)*17
  2 cx, cy, w, h, (0, 0, 0)*17
  3 cx, cy, w, h, (0, 0, 0)*17
  4 cx, cy, w, h, (0, 0, 0)*17
  .
  .
  .
  ```
- training
  * class `0`表示person，其他为关键点的小框
  * 上面第一行后面的(kx, ky, kc), `kx`, `ky`分别表示预测关键点的x,y坐标, `kc`不参与训练，只是作为数据标签处理与计算loss的一个筛选标志
  * 论文中网络输出维度为`3K+1`，这里的`K`=17，也就是关键点的个数，由于有个`person`类别，所以有个`+1`
  * 对于数据增强导致的出现在图片之外的关键点，`kc`会设置为0
  * 标签分配时也是和`yolov5`一样, 只是多返回一个关键点部分
  * 计算keypoint损失的时候只会计算标签中`kc` > 0的关键点
  * 关键点回归的范围是[-2, 2] * anchor
    + 由于keypoint学到的是关于中心点`cx`,`cy`的偏移，
    + 而yolov5回归bbox`w`,`h`的时候范围是[0, 4] * anchor，
    + 因此keypoint回归范围就是[-2, 2] * anchor
- inference
  * 预测的person框与**关键点框**可以一起处理，一起做nms，也可以分开处理，分开做nms
  * 只有预测为person类的才会有预测的**关键点**
  * 预测的**关键点**可以直接作为预测，也可以与**关键点框**一起融合矫正得到最后的关键点
    + 具体融合策略为：计算**关键点**与相关的**关键点框的中心点**的距离，如果距离小于预设的阈值则选取中心点作为新关键点，否则就保持原关键点
- thinking(maybe wrong)
  * 其实对于person keypoint的检测(人体姿态估计), 由于人体大小分布不一致，各个关节活动范围大，相对位置不固定，还存在位置、衣服遮挡等情况，场景更复杂(相比于face landmark), 所以采用类似与face landmark检测的pipeline(检测+关键点回归)，效果就不会理想。(参考[https://mp.weixin.qq.com/s/pRtckadSk34YPX2sjt4jSg](https://mp.weixin.qq.com/s/pRtckadSk34YPX2sjt4jSg), 该链接方法应该是**对行人进行裁剪下来之后再单独回归关键点**，虽然这样做或许丧失了检测网络中回归边框的信息, 但我依旧认为上面描述的观点)
  * 但是该方法其实不看检测关键点框的部分，就是检测+回归关键点的pipeline(且简单运行demo测试，发现不融合关键点与关键点框效果也可以), 这样的成功可能有以下几个原因：
    + yolov5本身性能较好
    + 高输入分辨率1280; DEKR、HigherHRNet等heatmap方法输入分辨率是512，640(centerNet下采样步长为4，能够检测+姿态估计，所以我觉得这个可能和1280的高分辨率有关系，毕竟相同的下采样步长，1280分辨率的到的特征图更大)
    + 回归关键点是基于框的中心点回归偏移, 感觉这个比较巧妙的与目标检测进行了结合(就比如说裁剪人再回归关键点，应该是直接回归关键点，而不是回归这个偏移(I guess))
