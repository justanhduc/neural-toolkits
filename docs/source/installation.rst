Installation
============

.. contents::
   :depth: 3
   :local:

Requirements
------------

Pytorch
^^^^^^^

`neural-toolkits` is built on top of Pytorch, so obviously Pytorch is needed.
Please refer to the official `Pytorch website <https://pytorch.org/>`_ for installation details.


Other dependencies
^^^^^^^^^^^^^^^^^^

In `neural-toolkits`, we use several backends to visualize training, so it is necessary to install
some additional packages. For convenience, installing the package will install all the required
dependencies. Optional dependencies can be installed as instructed below.


Install `neural-toolkits`
-------------------------

From Github
^^^^^^^^^^^

To install the bleeding-edge version, which is highly recommended, run ::

    pip install git+git://github.com/justanhduc/neural-toolkits.git@master

We also provide a version with some fancy Cuda/C++ implementations
that are implemented or collected from various sources. To install this version, run ::

    pip install neural-toolkits --cuda-ext

Uninstall `neural-toolkits`
---------------------------

Simply use pip to uninstall the package ::

    pip uninstall neural-toolkits

Why would you want to do that anyway?
