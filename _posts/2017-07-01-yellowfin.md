---
layout: post
title:  "The YellowFin optimizer: analysis and implementation in keras"
date:   2017-07-01
categories: ["science"]
---

YellowFin is a new type of optimizer for deep neural networks that is purported to achieve faster convergence and better results than state-of-the-art optimizers like Adagrad, RMSProp, and Adam in many domains.

Introduced by Jian Zhang, Ioannis Mitliagkas, and Christopher Ré in the paper [YellowFin and the Art of Momentum Tuning](https://arxiv.org/abs/1706.03471), this method is novel due to the tuning of the *momentum parameter*.