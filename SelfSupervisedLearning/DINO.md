- **paper**: [https://arxiv.org/pdf/2104.14294.pdf](https://arxiv.org/pdf/2104.14294.pdf) 
- **code**: [https://github.com/facebookresearch/dino](https://github.com/facebookresearch/dino) 

## 方法
- 将知识蒸馏运用到self-supervised learning中
- 其中`学生`与`老师`模型架构一样，并且在训练的时候反向传播更新`学生`模型，然后通过EMA更新`老师`模型(在线蒸馏)
![F2](../imgs/DINO/F2.png)
- 采用`multi-crop`生成不同的视图作为输入，其中包括两个`全局视图`x<sub>1</sub><sup>g</sup>和x<sub>2</sub><sup>g</sup>和几个`局部视图`;
- 然后最小化蒸馏loss：
![e3](../imgs/DINO/e3.png)
  * 其中H(a,b)=-a log b

- 避免崩塌
  * 使用centering和sharpening
- 整体流程伪代码：
![A1](../imgs/DINO/A1.png)

## 一些细节
- 模型的输出需要使用softmax归一化(`学生`模型为例)：
![e1](../imgs/DINO/e1.png)
- `老师`模型也是如此，不过是τ<sub>s</sub> → τ<sub>t</sub>
- 然后最小化蒸馏loss：
![e3](../imgs/DINO/e3.png)
  * 其中H(a,b)=-a log b

- centering可以理解为在`老师`中加了一个bias
  * g<sub>t</sub>(x) ← g<sub>t</sub> + c
  * 使用EMA更新center
![e4](../imgs/DINO/e4.png)
- sharpening则是在`老师`的softmax归一化的时候使用较低的τ<sub>t</sub>值
- 对于`multi-crop`
  * `全局视图`是裁剪224<sup>2</sup> 的分辨率，并且包含大部分原图信息(超过50%)
  * `局部视图`是裁剪96<sup>2</sup> 的分辨率，并且包含小部分原图信息(低于50%)
