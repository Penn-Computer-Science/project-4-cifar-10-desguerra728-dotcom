## Reflection

**What differences did you notice between MNIST and CIFAR-10?**
- The images from CIFAR-10 are much more complex, and the images within each class are more disimilar.
- It was much harder to build the model for it to reach higher accuracy values without overfitting.
- Since I had to increase the number of filters from 32 to 64 for CIFAR-10, the time per epoch increased significantly.

**What changes to your architecture helped performance?**
- Increasing the dimensions of the Conv2D increased validation accuracy.
- Increasing percent dropout generallly decreased the amount of overfit (but sometimes my graphs ended up oscillating).
- - I needed to balance out the amount of dropout to the layer 

**If you had more time, what improvements would you try next?**
- Testing specifically with the sequence of the layers; changing the order (I mostly just changed the amount of layers and their settings). 
- Changing the dimensions for the Conv2D to ones I haven't already tried.


## Charts
### CIRFAR-10
![CIFAR-10-Confusion-Matrix
](https://github.com/Penn-Computer-Science/project-4-cifar-10-desguerra728-dotcom/blob/c431a7bfc73bcde663b9eac632336dfff2ed1975/CIFAR-10-plots/confusion-matrix.png)

![CIFAR-10-Validation-Loss
](https://github.com/Penn-Computer-Science/project-4-cifar-10-desguerra728-dotcom/blob/f04475d6df48763002c20460765eea041491d838/CIFAR-10-plots/loss-and-accuracy.png)

### CIRFAR-100
![CIFAR-10-Confusion-Matrix
](https://github.com/Penn-Computer-Science/project-4-cifar-10-desguerra728-dotcom/blob/c431a7bfc73bcde663b9eac632336dfff2ed1975/CIFAR-10-plots/confusion-matrix.png)

![CIFAR-10-Validation-Loss
](https://github.com/Penn-Computer-Science/project-4-cifar-10-desguerra728-dotcom/blob/f04475d6df48763002c20460765eea041491d838/CIFAR-10-plots/loss-and-accuracy.png)
