**yolov5的标签分配：我认为是先fixed再adaptive**
1. 首先通过尺度scale，和offset分配好anchor(fixed);
2. 在回归bbox的时候是在用iou loss(CIOU), 在回归objectness的时候是采用CIOU值作为target,
	根据ciou值回归objectness来抑制一些质量差(CIOU低)anchor, 其实也可以看作是动态的根据CIOU值来抑制质量差的anchor;
3. 与adaptive相比，问题在于，虽然2可以抑制一些质量差(CIOU低)的anchor, 
	但可能会漏掉一些质量高的，但在1中未给出的anchor(不过yolov5的offset会在1分配大量的anchor给GT，或许一定程度能够缓解);
4. 还有就是，尽管objectness或许会抑制低质量的anchor，
	但网络在bbox回归依然会拉近该低质量anchor预测与GT的距离，这样还是会影响模型收敛;
5. 4其实也算是个矛盾点，即objectness认为某个anchor质量不好，但网络的bbox回归分支还在尽力回归该anchor,
	有可能会造成预测框location质量高，但confidence低的情况;


**adaptive assign**
1. 会在分配anchor之前，通常会计算cls loss和bbox loss, 再通过某个策略来根据loss进行anchor的分配;
2. 在计算objectness的时候也通常不需要通过一些例如CIOU的手段来筛选，
	因为在1步已经得到了模型认为的高质量anchor，既然都是高质量anchor，则也不需要进一步筛选，直接让网络拉满回归objectness;
3. 因为前一步anchor的筛选已经是根据cls loss和bbox loss选择的较好的anchor，所以就算objectness的回归是使用iou值，大概率该值也较大，
	且如果objectness对某anchor进行抑制，相当于cls和bbox认为这是个高质量anchor，但objectness认为是一个低质量anchor，可能会有回归冲突；


