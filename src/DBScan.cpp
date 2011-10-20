#include "DBScan.hpp"

namespace machine_learning
{
    DBScan::DBScan()
    {
        number_of_points = 0;
    }

    std::map<base::Point*, int> DBScan::scan(base::samples::Pointcloud* pointcloud, int min_pts, double epsilon)
    {

        int cluster_id = 0;
        number_of_points = pointcloud->points.size();

        // Initialize clustering
        initialize(pointcloud);

        // Scan for clusters
        for(int i = 0; i < number_of_points; i++) {
            base::Point* p = &pointcloud->points[i];
            if (clustering[p] == UNCLASSIFIED) {
                // Try to classify point
                if (expandCluster(p, cluster_id)) {
                    // New cluster found.
                    cluster_id++;
                }
            }
        }
        return clustering;
    }

    void DBScan::reset()
    {
        clustering.clear();
        //it = 0;
    }

    // All points in point cloud are being initialized as unclassified.
    void DBScan::initialize(base::samples::Pointcloud* pointcloud)
    {
        if(!clustering.empty()) {
            reset();
        }

        for(int i = 0; i < number_of_points; i++) {
            classify(&pointcloud->points[i], UNCLASSIFIED);
        }
    }

    bool DBScan::expandCluster(base::Point* start_point, int cluster_id)
    {

    }

    void DBScan::classify(base::Point* point, int cluster_id)
    {
        bool ret = clustering.insert(std::pair<base::Point*, int>(point, cluster_id)).second;
        std::cout << (ret ? "Inserted point " : "Point already inserted: ") << machine_learning::pointToString(point) << std::endl;
    }

}
