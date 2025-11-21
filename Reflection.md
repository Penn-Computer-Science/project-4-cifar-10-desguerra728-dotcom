## Reflection

**What differences did you notice between MNIST and CIFAR-10?**
- It was much harder to build the model for it to reach higher accuracy values without overfitting.
- Since I had to increase the number of filters from 32 to 64 for CIFAR-10, the time per epoch increased significantly.

**What changes to your architecture helped performance?**
- Increasing the dimensions of the Conv2D increased validation accuracy.
- Increasing percent dropout generallly decreased the amount of overfit (but sometimes my graphs ended up oscillating).

**If you had more time, what improvements would you try next?**
- I would try changing the order of some of my layers to see how different layers are 'related' to each other.
- test more dimensions for the Conv2D


## Charts
**CIRFAR-10**
![CIFAR-10-Confusion-Matrix
](https://github.com/Penn-Computer-Science/project-4-cifar-10-desguerra728-dotcom/blob/c431a7bfc73bcde663b9eac632336dfff2ed1975/CIFAR-10-plots/confusion-matrix.png)

![CIFAR-10-Validation-Loss
](https://github.com/Penn-Computer-Science/project-4-cifar-10-desguerra728-dotcom/blob/f04475d6df48763002c20460765eea041491d838/CIFAR-10-plots/loss-and-accuracy.png)

**CIRFAR-100**
