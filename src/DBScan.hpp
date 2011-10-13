#ifndef _MACHINE_LEARNING_DBSCAN_HPP_
#define _MACHINE_LEARNING_DBSCAN_HPP_

#include <iostream>
#include <vector>
//#include <base/samples/pointcloud>

namespace machine_learning
{
    struct DBScanResult {
        base::samples::Pointcloud pointcloud;
        int cluster_id;
    }
    
	class DBScan
	{
		public: 
            DBScan(base::samples::Pointcloud* pointcloud, int min_pts, double epsilon);
            std::vector<DBScanResult> scan();
            bool expandCluster(base::Point* start_point, int cluster_id);
			void welcome();
	};
    
;

} // end namespace machine_learning

#endif // _MACHINE_LEARNING_DBSCAN_HPP_
