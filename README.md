# NeeDrop: Self-supervised Shape Representation from Sparse Point Clouds using Needle Dropping

<p align="center">
<img src="./doc/banner.gif" />
</p>

by: [Alexandre Boulch](https://www.boulch.eu), [Pierre-Alain Langlois](https://imagine.enpc.fr/~langloip/index.html), [Gilles Puy](https://sites.google.com/site/puygilles/) and [Renaud Marlet](http://imagine.enpc.fr/~marletr/)

[Project page](https://www.boulch.eu/2021_3dv_needrop) - Paper - Arxiv - Blog - [Code](https://github.com/valeoai/NeeDrop)

## Abstract

There has been recently a growing interest for implicit shape representations. Contrary to explicit representations, they have no resolution limitations and they easily deal with a wide variety of surface topologies. To learn these implicit representations, current approaches rely on a certain level of shape supervision (e.g., inside/outside information or distance-to-shape knowledge), or at least require a dense point cloud (to approximate well enough the distance-to-shape). In contrast, we introduce NeeDrop, an self-supervised method for learning shape representations from possibly extremely sparse point clouds. Like in Buffon’s needle problem, we “drop” (sample) needles on the point cloud and consider that, statistically, close to the surface, the needle end points lie on opposite sides of the surface. No shape knowledge is required and the point cloud can be highly sparse, e.g., as lidar point clouds acquired by vehicles. Previous self-supervised shape representation approaches fail to produce good-quality results on this kind of data. We obtain quantitative results on par with existing supervised approaches on shape reconstruction datasets and show promising qualitative results on hard autonomous driving datasets such as KITTI.