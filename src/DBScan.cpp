#include "DBScan.hpp"
#include <cmath>

namespace machine_learning
{
    DBScan::DBScan(base::samples::Pointcloud* pointcloud, unsigned int min_pts, double epsilon, bool use_z)
    {
        this->pointcloud = pointcloud;
        this->min_pts = min_pts;
        this->epsilon = epsilon;
        this->use_z = use_z;
        number_of_points = 0;
        initialize();
    }

    std::map<base::Point*, int> DBScan::scan()
    {
        // Initialize clustering
        initialize();

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
        number_of_points = 0;
    }

    // All points in point cloud are being initialized as unclassified.
    void DBScan::initialize()
    {
        if(!clustering.empty()) {
            reset();
        }

        cluster_id = 0;
        number_of_points = pointcloud->points.size();

        for(int i = 0; i < number_of_points; i++) {
            classify(&pointcloud->points[i], UNCLASSIFIED);
        }
    }

    bool DBScan::expandCluster(base::Point* start_point, int cluster_id)
    {
        // Neighbor points in epsilon radius
        std::vector<base::Point*> seeds = neighbors(start_point);
        if (seeds.size() < min_pts) {
            // start_point has not enough neighbors in epsilon radius (is no "Kernobjekt").
            // Therefore it does not belong to any cluster.
            clustering[start_point] = NOISE;
            return false;
        }

        // New cluster found. The start point and all its neighbors belong to it. Assign cluster ID.
        clustering[start_point] = cluster_id;
        for(std::vector<base::Point*>::iterator it = seeds.begin(); it != seeds.end(); it++) {
            clustering[*it] = cluster_id;
        }

        // Search for density-reachable points.
        while(!seeds.empty()) {
            std::vector<base::Point*>::iterator currentPointIt = seeds.begin(); // TODO maybe improve choice using heuristic
            std::vector<base::Point*> currentSeeds = neighbors(*currentPointIt);
            if(currentSeeds.size() >= min_pts) {
                // Current Point has enough neighbors in epsilon radius.
                for(std::vector<base::Point*>::iterator it = currentSeeds.begin(); it != currentSeeds.end(); it++) {
                    if(clustering[*it] == UNCLASSIFIED || clustering[*it] == NOISE) {
                        // Point has no cluster ID so far.
                        if(clustering[*it] == UNCLASSIFIED) {
                            // Add to cluster if unclassified. If it were classified as NOISE, we would already know that it has not enough neighbors.
                            seeds.push_back(*it);
                        }
                        // Assign cluster ID.
                        clustering[*it] = cluster_id;
                    }
                }
            }
            seeds.erase(currentPointIt);

        }

        cluster_id++;
    }

    void DBScan::classify(base::Point* point, int cluster_id)
    {
        bool ret = clustering.insert(std::pair<base::Point*, int>(point, cluster_id)).second;
        std::cout << (ret ? "Inserted point " : "Point already inserted: ") << machine_learning::pointToString(point) << std::endl;
    }

    std::vector<base::Point*> DBScan::neighbors(base::Point* point) {
        std::vector<base::Point>::iterator it;
        std::vector<base::Point*> neighbors;
        for(it = pointcloud->points.begin(); it != pointcloud->points.end(); it++) {
            if(euclidean_distance(point, &(*it)) <= epsilon) {
                neighbors.push_back(&(*it));
            }
        }
        return neighbors;
    }

    double DBScan::euclidean_distance(base::Point* p1, base::Point* p2, bool use_z)
    {
        int dimensions = (use_z ? 3 : 2);

        double sum = 0;
        for(int dim = 0; dim < dimensions; dim++) {
            sum += pow(((*p1)[dim] - (*p2)[dim]), 2);
        }
        return sqrt(sum);
    }

}
