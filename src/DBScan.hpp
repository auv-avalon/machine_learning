#ifndef _MACHINE_LEARNING_DBSCAN_HPP_
#define _MACHINE_LEARNING_DBSCAN_HPP_

#include <base/samples/pointcloud.h>
#include <iostream>
#include <vector>
#include <map>


namespace machine_learning
{

    // Helpers
    static std::string pointToString(base::Point* p)
    {
        std::stringstream ss;
        ss << "(" << (*p)[0] << "," << (*p)[1] << "," << (*p)[2] << ")";
        return ss.str();
    }

    struct DBScanResult {
        base::samples::Pointcloud pointcloud;
        int cluster_id;
    };

    struct ClusteredPoint {
        base::Point* point;
        int cluster_id;
    };

	class DBScan
	{
		public:
            static const int UNCLASSIFIED = -1;
            static const int NOISE = -2;

            DBScan();
            std::map<base::Point*, int> scan(base::samples::Pointcloud* pointcloud, unsigned int min_pts, double epsilon, bool use_z = false);
            void reset();
            double euclidean_distance(base::Point* p1, base::Point* p2, bool use_z = false);

        private:
            int cluster_id;
            unsigned int min_pts;
            double epsilon;
            bool use_z;
            base::samples::Pointcloud* pointcloud;

            std::map<base::Point*, int> clustering;
            int number_of_points;

            void initialize(base::samples::Pointcloud* pointcloud, unsigned int min_pts, double epsilon, bool use_z);
            bool expandCluster(base::Point* start_point, int cluster_id);
            void classify(base::Point* point, int cluster_id);
            std::vector<base::Point*> neighbors(base::Point* point);

	};

} // end namespace machine_learning

#endif // _MACHINE_LEARNING_DBSCAN_HPP_
