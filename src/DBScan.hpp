#ifndef _MACHINE_LEARNING_DBSCAN_HPP_
#define _MACHINE_LEARNING_DBSCAN_HPP_

#include <sonar_detectors/SonarDetectorTypes.hpp>
#include <base/samples/pointcloud.h>
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <boost/foreach.hpp>


namespace machine_learning
{

    // Helpers
    static std::string pointToString(sonar_detectors::obstaclePoint& p)
    {
        std::stringstream ss;
        ss << "(" << p.position[0] << "," << p.position[1] << "," << p.position[2] << ")";
        return ss.str();
    }

	class DBScan
	{
		public:
            static const int FIRST_CLUSTER_ID = 0;
            static const int UNCLASSIFIED = -1;
            static const int NOISE = -2;

            DBScan(std::list<sonar_detectors::obstaclePoint>* featureList, unsigned int min_pts, double epsilon, bool use_z = false);
            std::map<sonar_detectors::obstaclePoint*, int> scan();
            void reset();
            double euclidean_distance(sonar_detectors::obstaclePoint& p1, sonar_detectors::obstaclePoint& p2, bool use_z = false);

            // Returns the amount of clusters found in the current point cloud.
            // WARNING: Call this method AFTER running scan()!
            int getClusterCount();

        private:
            int cluster_id;
            int cluster_count;
            unsigned int min_pts;
            double epsilon;
            bool use_z;
            std::list<sonar_detectors::obstaclePoint>* featureList;

            std::map<sonar_detectors::obstaclePoint*, int> clustering;
            int number_of_points;

            void initialize();
            bool expandCluster(sonar_detectors::obstaclePoint& start_point, int cluster_id);
            void classify(sonar_detectors::obstaclePoint& point, int cluster_id);
            std::vector<sonar_detectors::obstaclePoint*> neighbors(sonar_detectors::obstaclePoint& point);

	};

} // end namespace machine_learning

#endif // _MACHINE_LEARNING_DBSCAN_HPP_
