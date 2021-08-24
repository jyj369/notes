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
