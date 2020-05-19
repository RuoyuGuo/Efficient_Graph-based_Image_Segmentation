#import library
import numpy as np

from disjoint_set import disjoint_set
from itertools import product
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from skimage import filters, transform, io
from annoy import AnnoyIndex

def _preprocessing(img, sigma=0.8):
    return gaussian_filter(img, sigma)

##grid graph
def generate_graph(img, graph_type='grid', d=10, n_tree=10, search_k=-1):
    '''
    assume the image is channel last.
    generate grid graph or nearest neighbor graph
    
    return a list of sorted edge
    '''
    
    #Arguments:
    #img: A numpy array, channel last
    #graph_type
    #n_tree: larger tree, more precise
    
    img = _preprocessing(img)
    
    graphs = []
    rows = img.shape[0]
    cols = img.shape[1]
    num_vertices = rows * cols
    num_edges = (rows-1) * cols + (cols-1) * rows

    if graph_type == 'grid':
        #generate grid_graph
        for c in range(img.shape[2]):
            #create edges array 
            edges = np.empty((num_edges, 3), dtype=np.float64)

            #insert edge in edges array
            #we can see that each node connect to its right and down node 
            #except the most right columns don't have right connection
            #and the bottom row don't have down down connection
            index = 0
            for i in range(rows):
                for j in range(cols):

                    #add edge with current node to its right node
                    #if not the most right columns
                    if j < cols-1:
                        edges[index][0] = i*cols+j             
                        edges[index][1] = i*cols+j+1             
                        edges[index][2] = abs(img[i][j][c] - img[i][j+1][c])     #weight
                        index += 1

                    #add edge with current node to its down node
                    #if not the most right columns
                    if i < rows-1:
                        edges[index][0] = i*cols+j             
                        edges[index][1] = (i+1)*cols+j              
                        edges[index][2] = abs(img[i][j][c] - img[i+1][j][c])     #weight
                        index += 1  

            edges = edges[edges[:,2].argsort()]
            graphs.append(edges)

    elif graph_type == 'nn':
    #generate nearest neighbor graphs
    #using ANN to find 10 nearest neighbor of each pixel
    #using Annoy library
    
        f = 5
        t = AnnoyIndex(5, 'euclidean')
        nn_graph = []
        rows = img.shape[0]
        cols = img.shape[1]

        for i in range(rows):
            for j in range(cols):
                v = [img[i, j, 0], img[i, j, 1], img[i, j, 2] , i, j]
                t.add_item(i*cols+j, v)

        t.build(n_tree)

        for i in range(rows*cols):
            for neighbor in t.get_nns_by_item(i, d):
                if neighbor > i:
                    nn_graph.append([i, neighbor, t.get_distance(i, neighbor)])
                elif neighbor < i:
                    nn_graph.append([neighbor, i, t.get_distance(i, neighbor)])
                    
        nn_graph = np.array(nn_graph)
        nn_graph = nn_graph[np.unique(nn_graph[:, :2], axis=0, return_index=True)[1]]
        graphs.append(nn_graph[nn_graph[:,2].argsort()])
    
    else:
        raise ValueError('No such graph type, must be \'grid\' or \'grid\'')
    
    
    print('------------------------------------------')
    print(f'{graph_type} type graph construction done')
    print('------------------------------------------')
    return graphs

def segmentation(graph, rows, cols, k=300, graph_type='grid'):   
    '''
    take the output of generate_graph
    output is a disjoint_set, tells how to partition img
    
    return a disjoint_set object of your graph
    '''

    #Arguments:
    #grid_graphs: a nd numpy array, output of generate_gridgraph
    #rows = number of rows
    #cols = numer of columns
    #graph_type: grid or nn
    #k, threshold, larger k, larger component
    
    def _Tau(size, k):
        '''
        return threshold
        '''

        return k/size

    def _MINT(INT_C1, size_C1, INT_C2, size_C2, k):
        '''
        return minimum internal difference between two component
        '''

        return min(INT_C1 + _Tau(size_C1, k), INT_C2 + _Tau(size_C2, k))

    #n channel, n segmentations
    Segs_chs = []
    
    for ch in range(len(graph)):
    #construct segmentation of each channel
        if graph_type == 'grid':
            print('------------------------------------------')
            print(f'processing {ch}\'th channel of Grid Graph')
            print('------------------------------------------')
        elif graph_type == 'nn':
            print('----------------------------------')
            print(f'processing Nearest Neighbor Graph')
            print('----------------------------------')
        else:
            raise ValueError('No such graph type, must be \'grid\' or \'grid\'')
                
            
        seg = disjoint_set(rows*cols)

        for i in range(graph[ch].shape[0]):

            #find their set
            x = graph[ch][i][0]
            y = graph[ch][i][1]
            w = graph[ch][i][2]

            xp = seg.find(int(x))
            yp = seg.find(int(y))

            #not in the same set
            if xp != yp and w <= _MINT(seg.INT[xp], seg.arr[xp][2], seg.INT[yp], seg.arr[yp][2], k):
                seg.union(xp, yp)
                seg.update_INT(xp, yp, w)

        Segs_chs.append(seg)
    
    print('-----------------')
    print('Segmentation done')
    print('-----------------')
    return Segs_chs

def draw_img(segs, rows, cols, graph_type='grid'):
    '''
    colour img with respect to your segmentation
    
    return a coloured image, a numpy ndarray type
    '''

    #Arguments
    #segs: a list of disjoint_sets, output of segmentation
    #rows: rows in img
    #cols: cols in img
    #graph_type: 'grid' or 'nn', default grid
    
    for seg in segs:
        if not isinstance(seg, disjoint_set):
            raise TypeError('draw_img(segs, ...), segs must be a list-like of disjoint_sets')

    coloured_img = np.empty((rows, cols, 3), dtype=np.float64)
    con = None     
    
    if graph_type == 'grid':   
    #if two node are in same component in all segmentations
    #then they are in same component    
    
        con = disjoint_set(rows * cols)
     
        for i in range(rows):
            for j in range(cols-1):
                x = i * cols + j
                y = i * cols + j + 1
                if all([segs[ch].is_same_parent(x, y) for ch in range(len(segs))]):
                    con.union(x, y)

        for j in range(cols):
            for i in range(rows-1):
                x = i * cols + j
                y = (i+1) * cols + j
                if all([segs[ch].is_same_parent(x, y) for ch in range(len(segs))]):
                    con.union(x, y)

    elif graph_type == 'nn':
    #only need to conclude disjoint set 

        con = segs[0]
        
    else:
        raise ValueError('No such graph type, must be \'grid\' or \'grid\'')
    
    #colour img by component
    
    for i in range(rows):
        for j in range(cols):
            p = con.find(i*cols + j)
            np.random.seed(p)
            colour = np.random.randint(256, size=3)

            coloured_img[i][j] = colour
    
    '''
    omps_dict = con.conclusion()
    for key in comps_dict:
        colour = np.random.randint(256, size=3)
        for val in comps_dict[key]:
            x = val // cols
            y = val % cols
            coloured_img[x, y] = colour
    '''

    
    print('--------------------')
    print('colouring image done')
    print('--------------------')
    return coloured_img/255