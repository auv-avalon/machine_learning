#ifndef _MACHINE_LEARNING_CLUSTERING_UTILS
#define _MACHINE_LEARNING_CLUSTERING_UTILS

#include <base/eigen.h>
#include <vector>

namespace machine_learning
{

    static base::Vector3d getCentroid(std::vector<base::Vector3d*> &points) {
        int pointCount = points.size();
        base::Vector3d centroid;
        for(int i = 0; i < 3; i++) {
            int element_sum = 0;
            for(int j = 0; j < pointCount; j++) {
                element_sum += (*points[j])[i];
            }
            centroid[i] = (double) element_sum / (double) pointCount;
        }
        return centroid;
    }
    
}

#endif
