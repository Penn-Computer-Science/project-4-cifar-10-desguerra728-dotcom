# Reflection

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
---

# Charts
## CIRFAR-10
***Confusion Matrix***

![CIFAR-10-Confusion-Matrix
](https://github.com/Penn-Computer-Science/project-4-cifar-10-desguerra728-dotcom/blob/4c66931f18482922f20d94f0cec8ec9ca7b35e2b/CIFAR-10/Plots/confusion-matrix.png)

---
***Accuracy and Loss***

![CIFAR-10-Accuracy-Loss](https://github.com/Penn-Computer-Science/project-4-cifar-10-desguerra728-dotcom/blob/4c66931f18482922f20d94f0cec8ec9ca7b35e2b/CIFAR-10/Plots/loss-and-accuracy.png)

---
## CIFAR-100
***Confusion Matrix (Just Color)***

![CIFAR-100-Confusion-Matrix-Just-Color](https://github.com/Penn-Computer-Science/project-4-cifar-10-desguerra728-dotcom/blob/4c66931f18482922f20d94f0cec8ec9ca7b35e2b/CIFAR-100/Plots/Confusion%20Matrix%20Col.png)

---
***Confusion Matrix***

![CIFAR-100-Confusion-Matrix](https://github.com/Penn-Computer-Science/project-4-cifar-10-desguerra728-dotcom/blob/4c66931f18482922f20d94f0cec8ec9ca7b35e2b/CIFAR-100/Plots/Confusion%20Matrix%20Num.png)

---
***Accuracy and Loss***

![CIFAR-100-Accuracy-Loss](https://github.com/Penn-Computer-Science/project-4-cifar-10-desguerra728-dotcom/blob/4c66931f18482922f20d94f0cec8ec9ca7b35e2b/CIFAR-100/Plots/Loss%20Accuracy%20Plot.png)
