#!/bin/env python

##
## calculate the concave hull of a set of points
## see: CONCAVE HULL: A K-NEAREST NEIGHBOURS APPROACH
##      FOR THE COMPUTATION OF THE REGION OCCUPIED BY A
##      SET OF POINTS
##      Adriano Moreira and Maribel Yasmina Santos 2007
##
import sys
import os

dirpath = sys.path[0]
knn_path = os.path.join(dirpath,'..')
sys.path.insert(0,knn_path)


import numpy as np
import torch
from knn_cuda import KNN
import matplotlib.pyplot as plt
from matplotlib.path import Path
from th import lineintersect as li



def GetFirstPoint(dataset):
    ''' Returns index of first point, which has the lowest y value '''
    # todo: what if there is more than one point with lowest y?
    imin = torch.argmin(dataset[:,1])
    return dataset[imin]

def GetNearestNeighbors(dataset, point, k):
    point = point.view(1,-1)
    knn_tree = KNN(k, transpose_mode=True)
    distances, indices = knn_tree(dataset, point)
    distances, indices = torch.squeeze(distances),torch.squeeze(indices)

    return dataset[indices[:dataset.shape[0]]]

def SortByAngle(kNearestPoints, currentPoint, prevPoint):
    ''' Sorts the k nearest points given by angle '''
    angles =  torch.atan2(kNearestPoints[:,1]-currentPoint[1],
                      kNearestPoints[:,0]-currentPoint[0])- \
              torch.atan2(prevPoint[1] - currentPoint[1],
                          prevPoint[0] - currentPoint[0])
    angles = angles*180/np.pi
    angles = torch.fmod(angles+360,360)
    return kNearestPoints[torch.argsort(angles)]

def plotPoints(dataset):
    plt.plot(dataset[:,0],dataset[:,1],'o',markersize=10,markerfacecolor='0.75',
            markeredgewidth=1)
    plt.axis('equal')
    plt.axis([min(dataset[:,0])-0.5,max(dataset[:,0])+0.5,min(dataset[:,1])-0.5,
        max(dataset[:,1])+0.5])
    plt.show()

def plotPath(dataset, path):
    plt.plot(dataset[:,0],dataset[:,1],'o',markersize=10,markerfacecolor='0.65',
            markeredgewidth=0)
    path = np.asarray(path)
    plt.plot(path[:,0],path[:,1],'o',markersize=10,markerfacecolor='0.55',
            markeredgewidth=0)
    plt.plot(path[:,0],path[:,1],'-',lw=1.4,color='k')
    plt.axis('equal')
    plt.axis([min(dataset[:,0])-3,max(dataset[:,0])+3,min(dataset[:,1])-3,
        max(dataset[:,1])+3])

    plt.show()

def removePoint(dataset, point):
    delmask = torch.cuda.ByteTensor(dataset[:,0]!=point[0]) | torch.cuda.ByteTensor(dataset[:,1]!=point[1])
    newdata = dataset[delmask]
    return newdata


def concaveHull(dataset, k):
    assert k >= 3, 'k has to be greater or equal to 3.'
    begin = time.time()

    points = dataset
    # todo: remove duplicate points from dataset
    # todo: check if dataset consists of only 3 or less points
    # todo: make sure that enough points for a given k can be found

    assert type(points) == torch.Tensor, "You are stupid like Shaohui!!!!"
    for i in range(100):
        firstpoint = GetFirstPoint(points)
    hull = []
    # add first point to hull
    hull.append(firstpoint)
    # and remove it from dataset
    points = removePoint(points,firstpoint)
    currentPoint = firstpoint
    # set prevPoint to a Point righ of currentpoint (angle=0)
    prevPoint = (currentPoint[0]+10, currentPoint[1])
    step = 2
    while ( (not torch.equal(firstpoint, currentPoint) or (step==2)) and torch.prod(torch.tensor(points.shape)) > 0 ):
        if ( step == 5 ): # we're far enough to close too early
            points = torch.cat([points, firstpoint.view(1,-1)], dim=0)
        t = time.time()
        for i in range(100):
            kNearestPoints = GetNearestNeighbors(points, currentPoint, k)

        cPoints = SortByAngle(kNearestPoints, currentPoint, prevPoint)
        # avoid intersections: select first candidate that does not intersect any
        # polygon edge
        its = True
        i = 0
        while ( (its==True) and (i<cPoints.shape[0]) ):
                i=i+1
                if ( torch.equal(cPoints[i-1], firstpoint) ):
                    lastPoint = 1
                else:
                    lastPoint = 0
                j = 2
                its = False
                while ( (its==False) and (j<len(hull)-lastPoint) ):
                    its = li.doLinesIntersect(hull[step-1-1], cPoints[i-1],
                            hull[step-1-j-1],hull[step-j-1])
                    j=j+1
        if ( its==True ):
            print("all candidates intersect -- restarting with k = ",k+1)
            return concaveHull(dataset,k+1)
        prevPoint = currentPoint
        currentPoint = cPoints[i-1]
        # add current point to hull

        hull.append(currentPoint)
        points = removePoint(points,currentPoint)
        step = step+1
    # check if all points are inside the hull
    hull = [point.cpu().numpy()  for point in hull]
    p = Path(hull)
    dataset = dataset.cpu().numpy()
    pContained = p.contains_points(dataset, radius=0.0000000001)
    if (not pContained.all()):
        print("not all points of dataset contained in hull -- restarting with k = ",k+1)
        return concaveHull(dataset, k+1)

    print("finished with k = ",k)
    return hull




def test_concaveHull_1_k_5(points):
    points = torch.from_numpy(points).float()
    points = points.cuda()
    hull = concaveHull(points,5)
    return hull

if __name__ == '__main__':
    ### Teest DataSet

    points = np.array([[10, 9], [9, 18], [16, 13], [11, 15], [12, 14], [18, 12],
                       [2, 14], [6, 18], [9, 9], [10, 8], [6, 17], [5, 3],
                       [13, 19], [3, 18], [8, 17], [9, 7], [3, 0], [13, 18],
                       [15, 4], [13, 16]])

    points = torch.from_numpy(points).float().cuda()
    points_solution_k_5 = np.array([[3, 0], [10, 8], [15, 4], [18, 12], [13, 18], [13, 19],
                                    [9, 18], [6, 18], [3, 18], [2, 14], [9, 9], [5, 3], [3, 0]
                                    ])

    # points to test what happens if all points intersect
    points_intersect = np.array([[1, 1], [10, 3], [11, 8], [9, 14], [15, 21], [-5, 15], [-3, 10],
                                 [2, 5],  # from here the distracting points
                                 [9, 10], [8, 9], [8, 11], [8, 12], [9, 11], [9, 12]
                                 ])
    points_intersect_solution = np.array([[1, 1], [10, 3], [11, 8], [9, 14], [15, 21],
                                          [-5, 15], [-3, 10], [1, 1]
                                          ])

    points_E = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [6, 2], [6, 3], [5, 3], [4, 3],
                         [3, 3], [3, 4], [3, 5], [4, 5], [5, 5], [5, 6], [5, 7], [4, 7], [3, 7], [3, 8],
                         [3, 9], [4, 9], [5, 9], [6, 9], [6, 10], [6, 11], [5, 11], [4, 11], [3, 11], [2, 11],
                         [1, 11], [1, 10], [1, 9], [1, 8], [1, 7], [1, 6], [1, 5], [1, 4], [1, 3], [1, 2],
                         [5, 2], [4, 2], [3, 2], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
                         [2, 9], [2, 10], [3, 10], [4, 10], [5, 10], [3, 6], [4, 6], [5, 6], [4.5, 7], [3, 8.5],
                         ])
    # nmax = 800
    # alldata = np.random.randint(0, 5 * nmax, size=2 * nmax)
    # alldata = alldata.reshape(nmax, 2)
    # points = torch.from_numpy(alldata).float()
    import time
    begin = time.time()
    hull = test_concaveHull_1_k_5(points_intersect)
    print(time.time()-begin)
    print(hull)
    plotPath(points_intersect,hull)




