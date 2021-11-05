### Label Assign

**yolov5的标签分配：我认为是先fixed再adaptive**
1. 首先通过尺度scale，和offset分配好anchor(fixed);
2. 在回归bbox的时候是在用iou loss(CIOU), 在回归objectness的时候是采用CIOU值作为target,
	根据ciou值回归objectness来抑制一些质量差(CIOU低)anchor, 其实也可以看作是动态的根据CIOU值来抑制质量差的anchor,
	这个就类似GFL中class score回归直接采用iou作为target，有一定程度解决objectness与location预测不一致问题;
3. 与adaptive相比，问题在于，虽然2可以抑制一些质量差(CIOU低)的anchor, 
	但可能会漏掉一些质量高的，但在1中未给出的anchor(不过yolov5的offset会在1分配大量的anchor给GT，或许一定程度能够缓解);
4. 还有就是，尽管objectness或许会抑制低质量的anchor，
	但网络在bbox回归时CIOU loss较大，依然会尽力拉近该低质量anchor预测与GT的距离，这样还是会影响模型收敛;
5. 4其实也算是个矛盾点，即objectness认为某个anchor质量不好，但网络的bbox回归分支还在尽力回归该anchor,
	有可能会造成预测框location质量高，但confidence低的情况;


**adaptive assign(YOLOX)**
1. 会在分配anchor之前，通常会计算cls loss和bbox loss, 再通过某个策略来根据loss进行anchor的分配;
2. 在计算objectness的时候也通常不需要通过一些例如CIOU的手段来筛选，
	因为在1步已经得到了模型认为的高质量anchor，既然都是高质量anchor，则也不需要进一步筛选，直接让网络拉满回归objectness;
3. 因为前一步anchor的筛选已经是根据cls loss和bbox loss选择的较好的anchor，所以就算objectness的回归是使用iou值，大概率该值也较大，
	且如果objectness对某anchor进行抑制，相当于cls和bbox认为这是个高质量anchor，但objectness认为是一个低质量anchor，可能会有回归冲突；


**TOOD** 
1. 计算loss时与yolov5有点相似，通过class score与iou值算的一个指标metric，从而抑制一些较差的anchor，
	对这个指标normalization+max(iou), 作为cls回归的targets，其实这个metric就有点像yolov5中objectness loss的CIOU作为targets,
	但这个metric相当于这是一个分类和定位联合标示，这个作为置信度更有代表性;
2. 且TOOD在回归bbox的时候使用这个metric与IOU loss的乘积作为损失，比yolov5直接使用CIOU回归更好，
	就对于那些metric低的bbox降低他们的损失权重，更少的去关注这一部分低质量的这部分bbox回归,
	就缓解了yolov5里4中的矛盾点，使回归更加一致;
3. 至于label assign，TOOD会使用metric中最大的n个anchor分配给标签，即标签分配是动态分配，class score的回归标签也采用metric,
	由于这个框架没有objectness，所以有点不一样，这个回归class score应该会更好，就类似于GFL中的class score回归;


感觉YOLOX中可以尝试将objectness或者class score回归target改成iou值试一下;

- 其实GFL中将class score/objectness回归target改成iou值，
	iou/centerness预测分支与class score分支是分开回归的，但是做nms时又是联合到一起使用, 但其实增加的iou/centerness预测分支就代表着定位质量,
	由于定位质量只针对positive，所以可能会有negative定位质量很高，然后再与class score一起联合使用，会出问题,
	导致定位质量与置信度不匹配的问题，其实就相当于是处理置信度与定位质量之间的关系;
	这个不匹配问题是建立在：
	- 标签分配时，将某个低质量的anchor box/point分配给了某个GT，但是在回归class score/objectness时的targets是1，就导致了
		可能回归出来的的某个bbox定位质量很差，但class score/Objectness的分值很高，所以nms可能会去掉定位质量更好的bbox;
	- 所以其实可以理解为class score/objctness改成iou值，是为了与bbox质量挂钩，也就是去掉那些低质量的anchor(依据:低质量的anchor回归出来的pred与GT的iou应该更低);
	- 但是, 如果说在标签分配层面能够做的比较好，分配给GT的都是高质量的anchor，那其实class score/objectness是采用iou值还是1就区别不是很大了,
		可能会有一些网络收敛上的不同;
	- 或许可以根据将class score/objectness的回归targets改成iou值和1，分别实验，看差距来断定当前模型的标签分配策略是否较好;



### 匈牙利算法与OTA

**匈牙利算法** 

匈牙利算法是为了实现更多的匹配对，在最小化cost矩阵的情况下，实现最多的匹配对，要求矩阵rows与cols不一致；

example：
- cost matrix shape [a, b], where a > b, 匈牙利算法返回b个匹配对，使得cost最小；

```python
similarity = cosine_similarity(s1, s2)
row_ind, col_ind = linear_sum_assignment(1 - similarity)
```


**OTA** 

(算法中GT与anchor分别对应下述的a和b)

OTA是最优运输问题，通过最小的cost将一个概率分布转化为另一个概率分布；

example:
- cost matrix shape [a, b], 假设a > b, OTA将a分配给b，即b中某个元素可获得多个a中元素，使得全局cost最小；
- 假设a > b, OTA将a分配给b，尽管b中某个元素可获得多个a中元素, **但两个b不会获得同一个a中的元素**；
- 也可以指定每个b元素需求a的个数, **但指定总个数要与a的总个数相同**；

OTA部分代码
```python
# 选取topk个iou，(num_gt, k)
topk_ious, _ = torch.topk(ious * is_in_boxes.float(), self.top_candidates, dim=1)
# 初始化gt需求个数，+1表示bg, (num_gt + 1, )
mu = ious.new_ones(num_gt + 1)
# 将topk选取的iou求和来获取每个gt的dynamic k
mu[:-1] = torch.clamp(topk_ious.sum(1).int(), min=1).float()
# 因为标准OT问题，商品与需求数量要一致，所以将剩下的归为bg
mu[-1] = num_anchor - mu[:-1].sum()
# 初始化anchor，每个anchor为一个商品，每个初始化为1, (num_anchors, )
nu = ious.new_ones(num_anchor)
# 得到cost, (num_gt, num_anchor)
loss = torch.cat([loss, loss_cls_bg.unsqueeze(0)], dim=0)

# Solving Optimal-Transportation-Plan pi via Sinkhorn-Iteration.
# 计算运输距离
_, pi = self.sinkhorn(mu, nu, loss)

# Rescale pi so that the max pi for each gt equals to 1.
rescale_factor, _ = pi.max(dim=1)
pi = pi / rescale_factor.unsqueeze(1)

max_assigned_units, matched_gt_inds = torch.max(pi, dim=0)
```


**SimOTA**

SimOTA就是单纯的将cost最小的topk分配出来，再针对重复分配的元素，再选取一个cost最小的进行分配；

example:
- 即对于cost matrix shape [a, b], 假设a > b, SimOTA就是针对每一个b对应的cost，选取最小的topk个作为分配，再对于这其中重复分配到a中某元素的，再比较cost，选取cost最小的b元素得到这个a元素, 其他的b则舍弃该a元素；
- 相比于标准OTA，这样其实舍弃了一部分的anchor来分配；

```python
n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
# 选取iou最高的前n_candidate_k个, [n_gt, n_candidate_k]
topk_ious, _ =  torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
# 根据iou的总和来动态选择每个gt选择多少个positive， [n_gt, ]
dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
for gt_idx in range(num_gt):
    _, pos_idx = torch.topk(
        cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
    )
    matching_matrix[gt_idx][pos_idx] = 1.0

del topk_ious, dynamic_ks, pos_idx

# [n_pos_anchors, ], 每个positive分配到的gt总数
anchor_matching_gt = matching_matrix.sum(0)
# 对于那些同一个positive分到给到了多个gt的,
# 选取他们之间cost最小的gt匹配，其他的不匹配
if (anchor_matching_gt > 1).sum() > 0:
    # 找到每个positive对应cost最小的gt索引
    _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
    # 先将所有anchor_matching_gt > 1的matching重置
    matching_matrix[:, anchor_matching_gt > 1] *= 0.0
    # 再将cost最小的gt设置匹配
    matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
```

**Wasserstein Distance**
- 常见的有很多衡量概率分布差异的度量方式，比如total variation（TRPO推导里面有用到），还有经常被用到的KL散度。
- 相比于这些度量方式，Wasserstein距离有如下一些好处。
  * 能够很自然地度量离散分布和连续分布之间的距离；
  * 不仅给出了距离的度量，而且给出如何把一个分布变换为另一分布的方案；
  * 能够连续地把一个分布变换为另一个分布，在此同时，能够保持分布自身的几何形态特征；
- 最小化`Wasserstein Distance`得到最优运输的解；
- ` Sinkhorn algorithm`(也就是OTA中的最优运输算法)是`Wasserstein Distance`的一种特殊情况, 为的是更快的迭代；
- 详情见[http://www.stat.cmu.edu/~larry/=sml/Opt.pdf](http://www.stat.cmu.edu/~larry/=sml/Opt.pdf)

### Sample imbalance

**UMOP**
1. 对每个特征层级分别求loss再做均值，来平衡各个层级的loss
2. 可调节参数的Progressive focal loss也只能缓解某一层特征层的采样不平衡问题

我觉得应该是1和2一起才能有平衡各个层及之间的采样不平衡，首先1简单平衡了各个层及的loss与收敛，
影响了Progressive focal loss的参数(其参数是根据训练的情况来自动调节的)，再通过参数调节每层的平衡情况，
变相的影响各个特征层之间的平衡；

其实yolov5应该注意到了各个特征层级之间的不平衡，所以才会设置balance这个参数组去调节objectness的loss，
所以yolov5计算损失是按照特征层级来计算的，有些框架是按照batch这个维度来计算损失的；
mmdetection也是按照特征层的维度计算的；

