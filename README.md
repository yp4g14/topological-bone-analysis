# topological-bone-analysis

This code can be used for topological porosity analysis of greyscale images.
The method binarizes the images, cuts the images into patches and uses persistent homology with a signed Euclidean distance transform (SEDT) filtration.
Each pore appears as a single point in Quadrant 2 of H0, and bone regions surrounded by pores are the features in Quadrant 1 of H1.
Persistence statistics summarise each of these quadrants of the persistence diagrams explain the porous characteristics in patches of the images.

See example/example.ipynb for example use cases.
