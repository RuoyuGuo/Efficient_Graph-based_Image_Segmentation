### Efficient Graph-based Image segmentation
This project implement Image segmentation method proposed by [3].

### Example:

![alt text](https://github.com/RuoyuGuo/Efficient_Graph-based_Image_Segmentation/blob/master/data/nara.jpg?raw=true)

### Requirement
Annoy == 1.16.3
Numpy == 1.18.1
Python == 3.7.6
scipy == 1.4.1

### Library used:
* Annoy
* numpy
* matplotlib
* skimage
* scipy

It may be compatible on newer or older version.

### How to use:
data: example image for test
results: corresponding results of image in data
segmentation.py: module 
disjoint_set.py: disjoint_set object
efficient Graph-based Image segmentation.ipynb: my note

### python code example:
```python
import segmentation as seg    #import library
from skimage import transform
from matplotlib import pyptlot

path = '\data\eiffel.jpg'  #read image
img = plt.imread(path).astype(np.int32)
img = transform.resize(img, (320, 240), preserve_range=True)

grid_graphs = seg.generate_graph(img)
Segs = seg.segmentation(grid_graphs, img.shape[0], img.shape[1], k=300)
segmented_img= seg.draw_img(Segs, img.shape[0], img.shape[1])

plt.imshow(segmented_img)
```

### API:
```python
generate_graph(img, graph_type='grid', d=10, n_tree=10, search_k=-1)
```
Parameters: 
* img: a nd numpy array of image
* graph_type: {'grid', 'nn'}
    * 'grid': generate a grid graph of image
    * 'nn': generate a nearest neighbor graph of image
* d: fixed number of neighbors when use ```graph_type='nn'```
* n_tree: see Annoy API, https://github.com/spotify/annoy
* search_k: see Annoy API, https://github.com/spotify/annoy

Return:
* a list of sorted edge in graph

```python
segmentation(graph, rows, cols, k=300, graph_type='grid')
```

Parameters:
* graph: output of ```generated_graph(...)```
* rows: number of row of image(resized)
* cols: number of column of image(resized)
* k: threshold, larger k, larger component
* graph_type: same as ```generated_graph(...)```

Return:
* a disjoint_set object showing partition of image

```python
draw_img(segs, rows, cols, graph_type='grid')
```
Parameters:
* segs: output of ```segmentation(...)```
* rows: number of row of image(resized)
* cols: number of column of image(resized)
* graph_type: same as ```generated_graph(...)```

Return:
a nd numpy array segmented recoloured image. pixels in same component have same colour.

### NB:
When use grid graph, it is better to set larger k, say 300, when use nn graph, small k can give better results, try 200, 230...

Increasing n_tree could give good segmentation, default is 10, but run longer time.

### Discussion:
I encounter some ```RuntimeWarning: overflow encountered in longlong_scalars``` when I run ```draw_img(...)```. Error message tells me it is caused by ```self._arr[yp][2] += self._arr[xp][2]``` at line 73 in disjoint_set.py. Hope someone can help me to solve it.

I can only success run it on small scale image, say 320 * 240 pixel. When I run it on high resolution image, result is died.

Feel free to ask me if you have any questions or suggestions. 

### reference:
[0] https://algorithms.tutorialhorizon.com/disjoint-set-union-find-algorithm-union-by-rank-and-path-compression/

[1] https://algorithms.tutorialhorizon.com/disjoint-set-data-structure-union-find-algorithm/
<<<<<<< HEAD

[2] http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf
=======
[2] http://people.cs.uchicago.edu/~pff/papers/seg-ijcv.pdf
>>>>>>> 5515681ca8ec50142cd79af323e3b160cef9e3bb
