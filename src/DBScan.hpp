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

            DBScan();
            std::map<base::Point*, int> scan(base::samples::Pointcloud* pointcloud, int min_pts, double epsilon);
            void reset();

        private:
            std::map<base::Point*, int>::iterator it;
            std::map<base::Point*, int> clustering;
            int number_of_points;

            void initialize(base::samples::Pointcloud* pointcloud);
            bool expandCluster(base::Point* start_point, int cluster_id);
            void classify(base::Point* point, int cluster_id);

	};

} // end namespace machine_learning

#endif // _MACHINE_LEARNING_DBSCAN_HPP_
