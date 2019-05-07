
# Computational Neuroscience Project: Discrete Visual Region Decoding Models

# Background
One of the most prominent endeavors of modern cognitive and computational neuroscience is the search for a neural sensory code, a statistical model of how the brain represent sensory information. The derivation of such a model holds benefits not only to our understanding of neural computation, but also prosthetic development for sensory-impaired individuals. In the visual system, Bayesian decoding models of visual cortex activity have been proposed for natural image processing (Kay, Naselaris, Prenger, & Gallant, 2008). This latter model predicted individual voxel responses across V1, V2, and V3, and then used that pattern of activity to identify the stimulus image – with no regard to information flow across visual areas. Understanding this information flow, both computationally and structurally, may improve the accuracy algorithms for image classification and reconstruction, and provide insight into how visual encoding is distributed along the human visual cortex.

# Aim and Hypothesis

For a set of natural images, I am interested in establishing clusters of voxels in each visual region that are maximally predictive of a given category of natural image, followed by probing correlations between activity of maximally predictive clusters across areas. I am also interested in examining rough spatial connections between these clusters by examining associations across 3D coordinates. I hypothesize that associations across more similar semantic categories would also be more similar both in terms of strength and spatial localization.

# Data

To access the data use din this project, please consult the readme file in the data folder of this repository

# Analysis

The annotated analysis script for this project is vim1_regions_rf_analysis_dg4hd.py
