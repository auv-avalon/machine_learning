#ifndef _MACHINE_LEARNING_DBSCAN_HPP_
#define _MACHINE_LEARNING_DBSCAN_HPP_

#include <base/eigen.h>
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <boost/foreach.hpp>


namespace machine_learning
{

    // Helpers
    static std::string pointToString(base::Vector3d &p)
    {
        std::stringstream ss;
        ss << "(" << p[0] << "," << p[1] << "," << p[2] << ")";
        return ss.str();
    }

    /* This class implements the DBScan density based clustering algorithm for the specific use case of clustering obstacle points generated from sonar scan samples. For further information consult the following book:
     * Ester, M.; Sander, J.: Knowledge Discovery in Databases - Techniken und Anwendungen. 2000. Springer. Berlin.
     */
	class DBScan
	{
		public:
            static const int FIRST_CLUSTER_ID = 0;
            static const int UNCLASSIFIED = -1;
            static const int NOISE = -2;

            DBScan(std::list<base::Vector3d*>* featureList, unsigned int min_pts, double epsilon, bool use_z = false);
            std::map<base::Vector3d*, int> scan();
            void reset();
            static double euclidean_distance(base::Vector3d *p1, base::Vector3d *p2, bool use_z = false);

            // Returns the amount of clusters found in the current point cloud.
            // WARNING: Call this method AFTER running scan()!
            int getClusterCount();

            // Returns the amount of points in the current point cloud classified as NOISE.
            // WARNING: Call this method AFTER running scan()!
            int getNoiseCount();

        private:
            int cluster_id;
            int cluster_count;
            unsigned int min_pts;
            double epsilon;
            bool use_z;
            std::list<base::Vector3d*>* featureList;

            std::map<base::Vector3d*, int> clustering;
            int number_of_points;

            void initialize();
            bool expandCluster(base::Vector3d *start_point, int cluster_id);
            void classify(base::Vector3d *point, int cluster_id);
            std::vector<base::Vector3d*> neighbors(base::Vector3d *point);
	};

} // end namespace machine_learning

#endif // _MACHINE_LEARNING_DBSCAN_HPP_
