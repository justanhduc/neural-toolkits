.. _layers:
.. currentmodule:: neural_toolkits

=============================
:mod:`layers` -- Basic Layers
=============================

.. module:: neural_toolkits.layers
   :platform: Unix, Windows
   :synopsis: Basics layers in Deep Neural Networks
.. moduleauthor:: Duc Nguyen

This section describes all the backbone modules of Neuralnet-pytorch.

.. contents:: Contents
   :depth: 4

Abstract Layers
===============

Attributes
----------
The following classes equip the plain ``torch`` modules with more bells and whistles.
Also, some features are deeply integrated into Neuralnet-pytorch,
which enables faster and more convenient training and testing of your neural networks.

.. autoclass:: neural_toolkits.layers.abstract._LayerMethod
    :members:
.. autoclass:: neural_toolkits.layers.abstract.MultiSingleInputModule
.. autoclass:: neural_toolkits.layers.abstract.MultiMultiInputModule
.. autoclass:: neural_toolkits.layers.abstract.SingleMultiInputModule
.. autoclass:: neural_toolkits.layers.abstract.Eye

Extended Pytorch Abstract Layers
--------------------------------

.. autoclass:: neural_toolkits.layers.Module
.. autoclass:: neural_toolkits.layers.Sequential

Quick-and-dirty Layers
----------------------

.. autodecorator:: neural_toolkits.layers.wrapper
.. autoclass:: neural_toolkits.layers.Lambda

Common Layers
=============

Extended Pytorch Common Layers
------------------------------

.. autoclass:: neural_toolkits.layers.convolution.Conv2d
.. autoclass:: neural_toolkits.layers.convolution.ConvTranspose2d
.. autoclass:: neural_toolkits.layers.convolution.DepthwiseSepConv2d
.. autoclass:: neural_toolkits.layers.convolution.FC
.. autoclass:: neural_toolkits.layers.convolution.Softmax

Extra Layers
------------

.. autoclass:: neural_toolkits.layers.blocks.ConvNormAct
.. autoclass:: neural_toolkits.layers.blocks.FCNormAct
.. autoclass:: neural_toolkits.layers.blocks.ResNetBasicBlock2d
.. autoclass:: neural_toolkits.layers.blocks.ResNetBottleneckBlock2d
.. autoclass:: neural_toolkits.layers.blocks.StackingConv

Activation and Aggregation
--------------------------

.. autoclass:: neural_toolkits.layers.aggregation.Activation
.. autoclass:: neural_toolkits.layers.aggregation.Sum
.. autoclass:: neural_toolkits.layers.aggregation.SequentialSum
.. autoclass:: neural_toolkits.layers.aggregation.ConcurrentSum

Graph Learning Layers
---------------------

.. autoclass:: neural_toolkits.layers.points.GraphConv
.. autoclass:: neural_toolkits.layers.points.BatchGraphConv
.. autoclass:: neural_toolkits.layers.points.GraphXConv
