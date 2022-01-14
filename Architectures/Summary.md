
## Summary
- Transformer强大的原因或许不在于`self-attention`, 而是其设计的架构
- `self-attention`无非做的就是`token mixing`和`channel mixing`.
  * 生成`q`、`k`、`v`以及`q`\*`k`是`channel mixing`.
  * `attention`\*`v`是`token mixing`, 是没有跨通道信息的融合的.
  * 随后的FFN再进行`channel mixing`.
- `MLP-Mixer`与`ConvMixer`都证明不一定需要`self-attention`，前者采用`MLP`实现`token mixing`, 后者采用`DWconv`实现`token mixing`.(类似的工作还有很多)
- `ELSA`中将`self-attention`替换成`DWconv`, 在较低FLOPs下模型精度更高.
- 诸多论文证明:
  * **`self-attention`可以和`DWconv`等价，将自注意力层替换为`DWconv` 模块，性能也是非常类似的**
  * **`vit` 等`transformer`等算法成功的关键在于精心设计的`transformer`架构**
- `ConvNext`将`ResNet`一步一步改成`transformer`的架构，在使用全卷积的情况下精度超过了`swin transformer`, 验证了`Transformer`架构的强大.

## Reference
[https://www.zhihu.com/question/510965760/answer/2308125551](https://www.zhihu.com/question/510965760/answer/2308125551) 
[https://zhuanlan.zhihu.com/p/455086818](https://zhuanlan.zhihu.com/p/455086818)
