import sys
import pandas as pd
import numpy as np
import os

data, n, Eps, MinPts = sys.argv[1:5] #argument
n = int(n)
Eps = int(Eps)
MinPts = int(MinPts)

file_name = data
data = pd.read_csv(data, sep='\t')
data = data.values

import matplotlib.pyplot as plt

#plt.plot(data[:,1], data[:,2], 'bo', markersize=1)
#plt.show()

c_points = np.zeros((n,3))
point_type = np.dtype([('id','i4'), ('cluster','i4')])
points = np.full(data.shape[0], -2, dtype=point_type) #-2 : unvisited / -1: noise / 0~n : cluster #


c_index = 0 # cluster#

data = np.core.records.fromarrays(data.transpose(), names='id, x, y', formats='i4, f8, f8')
points['id'] = data['id']

def getDistances(p1, p2): #p1=point, p2=points
    d_type = ([('id','i4'), ('distance','f8')])
    distances = np.zeros(p2.shape[0], dtype=d_type)
    distances['id'] = p2['id']
    distances['distance'] = np.sqrt(np.power(p2['x'] - p1['x'],2) + np.power(p2['y'] - p1['y'],2))
    return distances
    

def getNeighbors(p):
    dist = getDistances(p, data[data != p])
    dist = dist[dist['distance'] <= Eps]
    return np.array([data[data['id'] == d['id']] for d in dist])


def scan(p, neighbors, index):
    size = 0
    if neighbors.size < MinPts: #noise or border point
        points[points['id'] == p['id']] = (p['id'], index) # noise or border point
        if index == -1: #noise
            return 0
        else: #border
            return 1
    else: #core point
        size += 1
        points[points['id'] == p['id']] = (p['id'], c_index) # set cluster
        for neighbor in neighbors:
            if points[points['id'] == neighbor['id']]['cluster'] < 0: #not visited or noise
                size += scan(neighbor, getNeighbors(neighbor),c_index)
        return size



def DBSCAN():
    global c_index
    cluster_size = []
    while -2 in points['cluster']: #모든 점을 다 순회했을 때까지
        pid = np.random.choice(points[points['cluster'] == -2]['id'], 1) #not visited points 중에서 하나 선택
        p = data[data['id'] == pid]
        size = scan(p, getNeighbors(p),-1)
        if size != 0:
            cluster_size.append(size)
            print(cluster_size[c_index])
            c_index += 1
        
    return np.array(cluster_size)

                
import sys
sys.setrecursionlimit(10000)
clusters = (-DBSCAN()).argsort()[:n]
print(clusters)
ids_in_clusters = [points[points['cluster'] == cluster]['id']
               for cluster in clusters]


points_in_clusters = [np.array([data[data['id'] == d] for d in ids_in_cluster])
                  for ids_in_cluster in ids_in_clusters]
#print(points_in_clusters[0])

import os
for i in range(n):
    with open(os.path.join(os.getcwd(),'test-3',file_name[0:-4]+'_cluster_'+str(i)+'.txt'),"w") as f:
        for point in points_in_clusters[i]:
            f.write(str(point['id'][0]) + '\n')
        

#plt.plot(points_in_clusters[0]['x'], points_in_clusters[0]['y'], 'bo', markersize=1)
#plt.plot(points_in_clusters[1]['x'], points_in_clusters[1]['y'], 'go', markersize=1)
#plt.plot(points_in_clusters[2]['x'], points_in_clusters[2]['y'], 'ro', markersize=1)
#plt.plot(points_in_clusters[3]['x'], points_in_clusters[3]['y'], 'co', markersize=1)
#plt.plot(points_in_clusters[4]['x'], points_in_clusters[4]['y'], 'mo', markersize=1)
#plt.plot(points_in_clusters[5]['x'], points_in_clusters[5]['y'], 'yo', markersize=1)
#plt.plot(points_in_clusters[6]['x'], points_in_clusters[6]['y'], 'o', color='lime', markersize=1)
#plt.plot(points_in_clusters[7]['x'], points_in_clusters[7]['y'], 'o', color='gold', markersize=1)
#plt.show()

#print(points)

