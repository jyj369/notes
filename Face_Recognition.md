## Loss

$L_{\text {Arcface }}=-\frac{1}{m} \sum_{i=1}^{m} \log \left(\frac{e^{s \cdot\left(\cos \left(\theta_{y_{i}}+t\right)\right)}}{e^{s \cdot\left(\cos \left(\theta_{y_{i}}+t\right)\right)}+\sum_{j=1, j \neq y_{i}}^{n} e^{s \cdot \cos \theta_{j}}}\right)$
