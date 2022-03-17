.. topological-bone-analysis documentation master file, created by
   sphinx-quickstart on Thu Nov 11 17:12:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Topological-bone-analysis
=========================
Overview
--------
This code uses persistent homology with a signed Euclidean distance transform
to analyse 2D image files, with a patch-based approach. 

This code can be used for topological porosity analysis of greyscale images. 
The method binarizes the images, cuts the images into patches and uses 
persistent homology with a signed Euclidean distance transform (SEDT) 
Each pore appears as a single point in Quadrant 2 of H0, and bone regions 
surrounded by pores are the features in Quadrant 1 of H1. 
Persistence statistics summarise each of these quadrants of the persistence 
diagrams explain the porous characteristics in patches of the images.

See example/example.ipynb for example use cases.

Installation
------------
topological bone analysis can be installed from pypi, the packaging index

``pip install topological-bone-analysis``

Contents
--------

.. toctree::
   :maxdepth: 2

   example/example
   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
