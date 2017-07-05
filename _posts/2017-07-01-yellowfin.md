---
layout: post
title:  "The YellowFin optimizer: analysis and implementation in keras"
date:   2017-07-01
categories: ["science"]
---

YellowFin is a new type of optimizer for deep neural networks that is purported to achieve faster convergence and better results than state-of-the-art optimizers like Adagrad, Adam, and RMSProp in many domains. I'm going to:

- quickly review the concept of optimizers in deep learning
- give a summary of YellowFin and the concepts behind it
- review the TensorFlow implementation code, and how I [**hacked together Keras compatibility**](https://github.com/nnormandin/YellowFin_Keras)
- test it on some more obscure data to see how it performs 'in the wild'
<br><br>

## optimization in deep learning

The choice of optimizer (and associated parameters) in the process of training a deep neural network plays a large role in the rate of convergence and the overall loss of the model. An optimizer that converges slowly is computationally costly, and one that gets stuck in local minima will prevent the parameters from being tuned optimally. The standard in deep learning is the family of *gradient descent* optimizers.

Generally speaking, given a model with \\(d\\) parameters \\(\theta \in \mathbb{R}^d\\) you need to minimize a loss function \\( J(\theta) \\). We can do this by calculating the gradient of the loss function with respect to our parameters, \\(\nabla_\theta J(\theta)\\), and then descending towards a lower value by moving in the opposite direction of the gradient. There's an excellent [**overview of gradient descent optimization algorithms**](http://sebastianruder.com/optimizing-gradient-descent/) by Sebastian Ruder that goes into great detail. He's updated it as of June 2017 to include *AdaMax* and *Nadam*.
<br><br>

## *YellowFin* and momentum tuning

Introduced by Jian Zhang, Ioannis Mitliagkas, and Christopher Ré in the paper [**YellowFin and the Art of Momentum Tuning**](https://arxiv.org/abs/1706.03471), this method is novel due to the tuning of the *momentum parameter*.

Most state-of-the-art methods store gradients from previous time steps with some decay parameter to form a 'momentum' term that is combined with the gradient in the current time step. However, the parameters for the momentum term (eg. \\(\beta_1\\) and \\(\beta_2\\) for the *Adam* optimizer) are static during training. Here are my key takeaways from the *YellowFin* paper:

- Recent [**literature**](https://arxiv.org/abs/1705.08292) has suggested that algorithms conducting element-wise gradient tuning yield marginal benefits compared to vanilla stochastic-gradient descent. Furthermore, models trained with these techniques actually generalize more poorly.
- In another recent [**paper**](https://arxiv.org/abs/1605.09774), Mitliagkas et al. showed that asynchronous training introduces 'momentum-like dynamics' into optimization (amplifying the momentum that has already been specified in the parameters of the chosen optimizer).
- The authors have found that tuning momentum as part of the optimizer parameter search improves convergence, but note that conducting a grid-search for the correct momentum and learning rates is challenging. Therefore, they propose *automatic momentum tuning*.
- The authors demonstrate that *momentum is robust to learning rate misspecification*. The source of this property is that the spectral radius of the momentum operator, \\(\rho(\mathbf{\mathit{A}}_t)\\) is constant in some subset of the hyperparameter space. The spectral radius of a matrix \\(\rho(\mathit{X})\\) is the largest absolute value of its eigenvectors.

	In fact (for a 'one dimensional strongly convex quadratic objective'), the spectral radius is constant at \\(\rho(\mathbf{\mathit{A}}_t) = \sqrt\mu\\) as long as

	$$(1-\sqrt\mu)^2/h \leq \alpha \leq (1+\sqrt\mu)^2 / h$$

	where \\(\mu\\) is the momentum parameter with learning rate \\(\alpha\\) and curvature \\(\mathit{h}\\). This means that as long as the learning rate falls within the 'robust region' identified in the inequality above, the asymptotic behavior will be the same. The authors point out that this allows them to focus on freely tuning the momentum parameter for optimality, given that the above condition is satisfied.

- The authors next extend the results regarding learning rate robustness to functions other than strongly smooth convex ones in order to demonstrate *robustness to curvature*. They define a 'Generalized condition number' or GCN, which is a measure of curvature along a scalar slice of a function. They derive optimal hyperparameters \\(\mu^\*\\) and \\(\alpha^\*\\), but do not provide a convergence rate guarnatee. Instead, they provide 'empirical evidence to support this intuition.'
- Using a noisy one-dimensional quadratic to model gradient descent with a noisy gradient (is in mini-batch descent), the authors develop tuning rules for the YellowFin optimizer on learning rate \\(\alpha\\) and momentum \\(\mu\\). The rules are dependent on an estimate of the model's distance to a local minima \\(\mathit{D}\\), an estimate for gradient variance \\(\mathit{C}\\), and the largest/smallest generalized curvatures \\(\mathit{h_{max}}\\) and \\(\mathit{h_{min}}\\).
	
	At each step, the authors solve what they call the *'SingleStep'* equation, which is:

	$$ \mu_t,\alpha_t = \underset{\mu}{\mathrm{argmin}}( \mu D^2 + \alpha^2C) $$

	subject to:

	$$ \mu \geq (\frac{\sqrt{h_{max}/h_{min}}-1} {\sqrt{h_{max}/h_{min}}+1})^2 $$

	and:

	$$\alpha = \frac{(1-\sqrt\mu)^2}{h_{min}}$$

- Using these rules, the authors demonstrate evidence of superior performance on CIFAR100 on ResNet:

![results image](/assets/images/yellowfin_result.png)



- As a last (but I think important) note, the authors assume a negative log-probability objective function in their calculations for the measurement functions used as inputs for *SingleStep*. This means it should play nicely with normal classification loss functions (eg. binary and categorical cross-entropy), but probably not for other ones. I'll explore this later.

<br><br>

# making the TensorFlow version work in Keras

The authors of the YellowFin paper released an [**implementation**](https://github.com/JianGoForIt/YellowFin) of their work in TensorFlow at the time of publication, and [**another**](https://github.com/JianGoForIt/YellowFin_Pytorch) in PyTorch soon afterwards.

I was able to easily drop the `YFOptimizer` object into my TensorFlow projects, but noticed when I tried to use it in keras with the `TFOptimizer` wrapper that the methods were not compatible.

Digging into the source code at [**keras/optimizers**](https://github.com/fchollet/keras/blob/59cd1c3994153a66084b00fadcafad2af5a15dd7/keras/optimizers.py#L599-L628), I was quickly able to see that this class defines a `get_updates` function that requires the TensorFlow optimzer being wrapped to have explicit `compute_gradients` and `apply_gradients` attributes. Looking into the the YellowFin [**source**](https://github.com/JianGoForIt/YellowFin/blob/master/tuner_utils/yellowfin.py), you can see that the authors simplified the optimization process by aggregating  `compute_gradients` and `apply_gradients` into a single `minimize` function.

It's also worth noting that the YellowFin optimizer itself appears to be a wrapper for the `tf.train.MomentumOptimizer` object, which becomes an attribute of the `YFOptimizer`. This saves the authors a significant amount of work, and allows them to devote most of their code to the functions that approximate the distance to a local minima \\(\mathit{D}\\), an estimate for gradient variance \\(\mathit{C}\\), and the largest/smallest generalized curvatures \\(\mathit{h_{max}}\\) and \\(\mathit{h_{min}}\\).

I was able to make a slightly modified version of the YellowFin optimizer that seems to play nice with Keras by explicitly defining the `compute_gradients` method and passing the `global_steps` argument from the Keras API. I'd really like to make this a native keras optimizer from scratch, but right now it can be used in its slightly modified state like this:

```python
from yellowfin import YFOptimizer
from keras.optimizers import TFOptimizer

# define your optimizer
opt = TFOptimizer(YFOptimizer())

# compile a classification model
model.compoile(loss = 'categorical_crossentropy',
               metrics = ['acc'],
               optimizer = opt)
```

I've tested it on GPU and CPU with Ubuntu 16.04, but if you find any issues please don't hesitate to [**contact me**](/contact) or contribute to it on [**github**](https://github.com/nnormandin/yellowfin_keras). 

<br><br>

# benchmark: financial time series forecasting

Using the CIFAR10 deep CNN example from the Keras example repository, I was able to make a [**working example**](https://github.com/nnormandin/YellowFin_Keras/blob/master/examples/cifar10_cnn.py) of my modified YellowFin in Keras.

This is great, but I think it's useful to validate new tools on data sets that aren't part of the normal machine learning milieu. I'm playing around with some financial time series data right now (using 100 days of OHLCV data to forecast returns using hybrid recurrent/self normalizing networks), and I'm having a lot of trouble tuning the learning rate to properly train my models. Hopefully YellowFin is the answer I've been looking for.

