#include "DBScan.hpp"
#include <cmath>

namespace machine_learning
{
    DBScan::DBScan(std::list<base::Vector3d*>* featureList, unsigned int min_pts, double epsilon, bool use_z)
    {
        this->featureList = featureList;
        this->min_pts = min_pts;
        this->epsilon = epsilon;
        this->use_z = use_z;
        cluster_count = 0;
        number_of_points = 0;
        //initialize(); TODO Do we need this? Interferes with initialize() call in scan()
    }

    std::map<base::Vector3d*, int> DBScan::scan()
    {
        // Initialize clustering
        initialize();

        // Scan for clusters
        for(std::list<base::Vector3d*>::iterator it = featureList->begin(); it != featureList->end(); it++) {
            if (clustering[*it] == UNCLASSIFIED) {
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

        for(std::list<base::Vector3d*>::iterator it = featureList->begin(); it != featureList->end();  it++) {
            classify(*it, UNCLASSIFIED);
        }
    }

    bool DBScan::expandCluster(base::Vector3d *start_point, int cluster_id)
    {
        // Neighbor points in epsilon radius
        std::vector<base::Vector3d*> seeds = neighbors(start_point);
        if (seeds.size() < min_pts) {
            // start_point has not enough neighbors in epsilon radius (is no "Kernobjekt").
            // Therefore it does not belong to any cluster.
            clustering[start_point] = NOISE;
            return false;
        }

        // New cluster found. The start point and all its neighbors belong to it. Assign cluster ID.
        clustering[start_point] = cluster_id;
        for(std::vector<base::Vector3d*>::iterator it = seeds.begin(); it != seeds.end(); it++) {
            clustering[*it] = cluster_id;
        }

        while(!seeds.empty()) {
            const int PICK_POS = 0; // TODO maybe improve choice using heuristic
            base::Vector3d* seed = seeds[PICK_POS];
            std::vector<base::Vector3d*> currentSeeds = neighbors(seed);

            if(currentSeeds.size() >= min_pts) {
                // Current Point has enough neighbors in epsilon radius.
                BOOST_FOREACH( base::Vector3d* currentSeed, currentSeeds ) {
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

    void DBScan::classify(base::Vector3d *point, int cluster_id)
    {
        bool ret = clustering.insert(std::pair<base::Vector3d*, int>(point, cluster_id)).second;
        //std::cout << (ret ? "Inserted point " : "Point already inserted: ") << machine_learning::pointToString(point) << std::endl;
    }

    std::vector<base::Vector3d*> DBScan::neighbors(base::Vector3d *point) {
        std::list<base::Vector3d*>::iterator it;
        std::vector<base::Vector3d*> neighbors;
        for(it = featureList->begin(); it != featureList->end(); it++) {
            if(euclidean_distance(point, *it) <= epsilon) {
                neighbors.push_back(*it);
            }
        }
        return neighbors;
    }

    double DBScan::euclidean_distance(base::Vector3d *p1, base::Vector3d *p2, bool use_z)
    {
        int dimensions = (use_z ? 3 : 2);

        double sum = 0;
        for(int dim = 0; dim < dimensions; dim++) {
            sum += pow(( (*p1)[dim] - (*p2)[dim]), 2);
        }
        return sqrt(sum);
    }

    int DBScan::getClusterCount()
    {
        return cluster_count;
    }

    int DBScan::getNoiseCount()
    {
        int nc = 0;
        for(std::map<base::Vector3d*, int>::iterator it = clustering.begin(); it != clustering.end(); it++) {
            if(it->second == NOISE) {
                nc++;
            }
        }
        return nc;
    }

}
