## Loss Function
### Softmax
$L_{S}=-\frac{1}{m} \sum_{i=1}^{m} \log \left(\frac{e^{W_{y_{i}}^{T} x_{i}+b_{y_{i}}}}{\sum_{j=1}^{n} e^{W_{j}^{T} x_{i}+b_{j}}}\right)$

![s1](./imgs/Face_Recognition/softmax.png) 
![s2](./imgs/Face_Recognition/softmax2.png) 

传统的Softmax仍存在着很大的类内距离, 这种方式主要考虑样本是否能正确分类，缺乏类内和类间距离的约束。


### Center loss
$L=L_{S}+L_{C}=-\frac{1}{m} \sum_{i=1}^{m} \log \left(\frac{e^{W_{y_{i}}^{T} x_{i}+b_{y}}}{\sum_{j=1}^{n} e^{W_{j}^{T} x_{i}+b_{j}}}\right)+\frac{\lambda}{2} \sum_{i=1}^{m}\left\|x_{i}-c_{y_{i}}\right\|^{2}$

![c3](./imgs/Face_Recognition/centerloss.png) 
- Center Loss的整体思想是希望一个batch中的每个样本的feature离feature 的中心的距离的平方和要越小越好，
也就是类内距离要越小越好。
- Center Loss考虑到了使得类内紧凑，却不能使类间可分

### A-Softmax Loss(SphereFace)
Center Loss考虑到了使得类内紧凑，却不能使类间可分，而Contrastive Loss、Triplet Loss增加了时间上的消耗
在Softmax中$W^{T} x=\|W\| \cdot\|x\| \cdot \cos \theta$可得到：
$L_{S o f t m a x}=-\frac{1}{m} \sum_{i=1}^{m} \log \left(\frac{e^{\left\|W y_{i}\right\| \cdot\left\|x_{i}\right\| \cdot \cos \theta_{y_{i}}+b_{y_{i}}}}{\sum_{j=1}^{n} e^{\left\|W_{j}\right\| \cdot\left\|x_{i}\right\| \cdot \cos \theta_{j}+b_{j}}}\right)$



### Cosine Margin Loss


### Angular Margin Loss(ArcFace)


### Circle loss


### MagFace


