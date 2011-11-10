#include "DBScan.hpp"
#include <cmath>

namespace machine_learning
{
    DBScan::DBScan(std::list<sonar_detectors::obstaclePoint>* featureList, unsigned int min_pts, double epsilon, bool use_z)
    {
        this->featureList = featureList;
        this->min_pts = min_pts;
        this->epsilon = epsilon;
        this->use_z = use_z;
        cluster_count = 0;
        number_of_points = 0;
        std::cout << "DBScan parameters:\n" << "Epsilon: " << this->epsilon
            << "\nMin_Pts: " << this->min_pts << std::endl;
        //initialize(); TODO Do we need this? Interferes with initialize() call in scan()
    }

    std::map<sonar_detectors::obstaclePoint*, int> DBScan::scan()
    {
        // Initialize clustering
        initialize();

        // Scan for clusters
        for(std::list<sonar_detectors::obstaclePoint>::iterator it = featureList->begin(); it != featureList->end(); it++) {
            if (clustering[&(*it)] == UNCLASSIFIED) {
                // Try to classify point
                if (expandCluster(*it, cluster_id)) {
                    // New cluster found.
                    cluster_id++;
                    cluster_count++;
                }
            }
        }
        return clustering;
    }

    void DBScan::reset()
    {
        clustering.clear();
        number_of_points = 0;
        cluster_count = 0;
        cluster_id = FIRST_CLUSTER_ID;
    }

    // All points in point cloud are being initialized as unclassified.
    void DBScan::initialize()
    {
        if(!clustering.empty()) {
            reset();
        }

        cluster_id = FIRST_CLUSTER_ID;
        number_of_points = featureList->size();

        for(std::list<sonar_detectors::obstaclePoint>::iterator it = featureList->begin(); it != featureList->end();  it++) {
            classify(*it, UNCLASSIFIED);
        }
    }

    bool DBScan::expandCluster(sonar_detectors::obstaclePoint& start_point, int cluster_id)
    {
        // Neighbor points in epsilon radius
        std::vector<sonar_detectors::obstaclePoint*> seeds = neighbors(start_point);
        if (seeds.size() < min_pts) {
            // start_point has not enough neighbors in epsilon radius (is no "Kernobjekt").
            // Therefore it does not belong to any cluster.
            clustering[&start_point] = NOISE;
            return false;
        }

        // New cluster found. The start point and all its neighbors belong to it. Assign cluster ID.
        clustering[&start_point] = cluster_id;
        for(std::vector<sonar_detectors::obstaclePoint*>::iterator it = seeds.begin(); it != seeds.end(); it++) {
            clustering[*it] = cluster_id;
        }

        while(!seeds.empty()) {
            const int PICK_POS = 0; // TODO maybe improve choice using heuristic
            sonar_detectors::obstaclePoint* seed = seeds[PICK_POS];
            std::vector<sonar_detectors::obstaclePoint*> currentSeeds = neighbors(*seed);

            if(currentSeeds.size() >= min_pts) {
                // Current Point has enough neighbors in epsilon radius.
                BOOST_FOREACH( sonar_detectors::obstaclePoint* currentSeed, currentSeeds ) {
                    if(clustering[currentSeed] == UNCLASSIFIED || clustering[currentSeed] == NOISE) {
                        // Point has no cluster ID so far.
                        if(clustering[currentSeed] == UNCLASSIFIED) {
                            // Add to cluster if unclassified. If it were classified as NOISE, we would already know that it has not enough neighbors.
                            seeds.push_back(currentSeed);
                        }
                        // Assign cluster ID.
                        clustering[currentSeed] = cluster_id;
                    }
                }
            }

            if(seeds.begin() != seeds.end()) {
                seeds.erase(seeds.begin()+PICK_POS);
            }

        }

        cluster_id++;
        return true;
    }

    void DBScan::classify(sonar_detectors::obstaclePoint& point, int cluster_id)
    {
        bool ret = clustering.insert(std::pair<sonar_detectors::obstaclePoint*, int>(&point, cluster_id)).second;
        std::cout << (ret ? "Inserted point " : "Point already inserted: ") << machine_learning::pointToString(point) << std::endl;
    }

    std::vector<sonar_detectors::obstaclePoint*> DBScan::neighbors(sonar_detectors::obstaclePoint& point) {
        std::list<sonar_detectors::obstaclePoint>::iterator it;
        std::vector<sonar_detectors::obstaclePoint*> neighbors;
        for(it = featureList->begin(); it != featureList->end(); it++) {
            if(euclidean_distance(point, *it) <= epsilon) {
                neighbors.push_back(&(*it));
            }
        }
        return neighbors;
    }

    double DBScan::euclidean_distance(sonar_detectors::obstaclePoint& p1, sonar_detectors::obstaclePoint& p2, bool use_z)
    {
        int dimensions = (use_z ? 3 : 2);

        double sum = 0;
        for(int dim = 0; dim < dimensions; dim++) {
            sum += pow((p1.position[dim] - p2.position[dim]), 2);
        }
        return sqrt(sum);
    }

    int DBScan::getClusterCount()
    {
        return cluster_count;
    }

}
