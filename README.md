# Megapixel Resolution Training

As described by:

http://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf

Adapted from initial DCGAN code:

https://github.com/carpedm20/DCGAN-tensorflow


Structure of GAN can be varied from 8x8 to 512x512 and onwards. Everything above 8x8 will try to load a model trained on the previous resolution.

```python
python main.py 8
python main.py 16
...
python main.py 512
```

[========== Original README ==========]

# DCGAN in Tensorflow

Tensorflow implementation of [Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434) which is a stabilize Generative Adversarial Networks. The referenced torch code can be found [here](https://github.com/soumith/dcgan.torch).

![alt tag](DCGAN.png)

* [Brandon Amos](http://bamos.github.io/) wrote an excellent [blog post](http://bamos.github.io/2016/08/09/deep-completion/) and [image completion code](https://github.com/bamos/dcgan-completion.tensorflow) based on this repo.
* *To avoid the fast convergence of D (discriminator) network, G (generator) network is updated twice for each D network update, which differs from original paper.*

