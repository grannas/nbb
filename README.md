# Neural Best Buddies
Algorithm implemented based on [Neural Best-Buddies: Sparse Cross-Domain Correspondence](https://arxiv.org/pdf/1805.04140.pdf) for feature matching between images.

Neural activations ("feature maps") in different layers are used to match semantic correspondences between images.

<img src="/examples/activation0.png" width="150"/> <img src="/examples/activation1.png" width="150"/> <img src="/examples/activation2.png" width="150"/> <img src="/examples/activation3.png" width="150"/> <img src="/examples/activation4.png" width="150"/>

The end result is matched pixels in two images.

<img src="/examples/dog.png" width="150"/> <img src="/examples/cat.png" width="150"/>  <img src="/examples/king.png" width="150"/> <img src="/examples/slayer.png" width="150"/>

By adding a constraint where to initially look inside the images, you can find correspondences in the same image.

<img src="/examples/double.png" width="150"/>

The algorithm relies heavily on geometry and is therefore easily fooled.

<img src="/examples/fish1.png" width="150"/> <img src="/examples/fish2.png" width="150"/>

## References
* Kfir Aberman, Jing Liao, Mingyi Shi, Dani Lischinski, Daniel Cohen-Or, Chen Baoquan. Neural Best-Buddies: Sparse Cross-Domain Correspondence, ACM Transactions on Graphics (TOG), 37(4), 2018.
* X. Huang and S. Belongie, "Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization," ICCV, 2017, pp. 1510-1519.
