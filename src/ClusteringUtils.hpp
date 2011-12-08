#ifndef _MACHINE_LEARNING_CLUSTERING_UTILS
#define _MACHINE_LEARNING_CLUSTERING_UTILS

#include <base/eigen.h>
#include <vector>

namespace machine_learning
{

	/* Returns the centroid of the submitted pointcloud. A centroid is a
	 * point whose components are the mean of the pointcloud point's components.
	 * You could think of it as "center of gravity".
	 *
	 * This algorithm is based upon the following book:
	 * "Ester, M.; Sander, J.: Knowledge Discovery in Databases - Techniken
	 * und Anwendungen. 2000. Springer. Berlin."
	 */
    static base::Vector3d centroid(std::vector<base::Vector3d*> &points) {
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
    
    /*
     * Returns a pretty print of the 3D Vector (no linefeed).
     */
    static std::string pointToString(base::Vector3d &p)
    {
        std::stringstream ss;
        ss << "(" << p[0] << "," << p[1] << "," << p[2] << ")";
        return ss.str();
    }

}

#endif
