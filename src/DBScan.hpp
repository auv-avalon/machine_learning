#ifndef _MACHINE_LEARNING_DBSCAN_HPP_
#define _MACHINE_LEARNING_DBSCAN_HPP_

#include <base/eigen.h>
#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <boost/foreach.hpp>
#include <cmath>

namespace machine_learning
{
    /* This class implements the DBScan density based clustering algorithm for the specific use case of clustering obstacle points generated from sonar scan samples. For further information consult the following book:
     * Ester, M.; Sander, J.: Knowledge Discovery in Databases - Techniken und Anwendungen. 2000. Springer. Berlin.
     * See also: http://en.wikipedia.org/wiki/DBSCAN
     * 
     * T is the type of points (eg. base::Vector3d), which should be clustered. 
     * The norm() and the subtraktion (-) needs to be defined for type T
     */    
    template < typename T>
    class DBScan
    {
	public:
		static const int FIRST_CLUSTER_ID = 0;
		static const int UNCLASSIFIED = -1;
		static const int NOISE = -2;

		/* This constructor sets up the whole DBScan (DB refers to "density based") clustering
		 * environment. The featureList can be understood as a point cloud in the R^3 vector
		 * space. Roughly, a cluster is defined by a "core point" ("Kernobjekt") having min_pts
		 * points in its epsilon radius (self excluded). The cluster can be expanded, if points
		 * in that radius are core points themselves. For the details, please refer to the mentioned
		 * literature.
		 * There are the following configuration parameters:
		 * @param min_pts 	Minimum point count needed in epsilon radius of a point p to define p as
		 *                	core point.
		 * @param epsilon 	Radius from point p in which's range have to be min_pts points to define
		 *                	p as core point.
		 * @param use_z   	This flag defines whether to interpret the vector as 2D or 3D. If use_z is false, only the
		 *                	X and Y coordinates are concerned. This is equal to (X,Y,0).
		 * @param distance	Alternitive distance-function between two points
		 * 			If 0, the euklidian-distance is used
		 */
		DBScan(std::list<T*>* featureList, unsigned int min_pts, double epsilon, bool use_dynamic_epsilon = false, double dynamic_epsilon_weight = 1.0, double (*distance)(T*,T*) = 0, bool speedup = false );

		/* Scans the pointcloud for clusters.
		 */
		std::map<T*, int> scan();
		static double euclidean_distance(T *p1, T *p2);

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
		bool use_dynamic_epsilon;
		double dynamic_epsilon_weight;
		std::list<T*>* featureList;

		std::map<T*, int> clustering;
		int number_of_points;
		bool speedup;
		typename std::list<T*>::iterator start_point;

		void initialize();
		void reset();
		bool expandCluster(T *start_point, int cluster_id);
		void classify(T *point, int cluster_id);
		std::vector<T*> neighbors(T *point);
		
		//Function-pointer to a the distance-function
		double (*pDistance)(T*,T*);
		
    };
    
  
    template < typename T>
    DBScan<T>::DBScan(std::list<T*>* featureList, unsigned int min_pts, double epsilon, bool use_dynamic_epsilon, double dynamic_epsilon_weight, double (*distance)(T*,T*), bool speedup)
    {
        this->featureList = featureList;
        this->min_pts = min_pts;
        this->epsilon = epsilon;
        this->use_dynamic_epsilon = use_dynamic_epsilon;
        this->dynamic_epsilon_weight = dynamic_epsilon_weight;
	this->speedup = speedup;
	
	if(distance == 0){
	  pDistance = euclidean_distance;
	}else{
	  pDistance = distance;
	}
	  
        cluster_count = 0;
        number_of_points = 0;
        //initialize(); TODO Do we need this? Interferes with initialize() call in scan()
    }

    template < typename T>
    std::map<T*, int> DBScan<T>::scan()
    {
        // Initialize clustering
        initialize();

        // Scan for clusters
        for(typename std::list<T*>::iterator it = featureList->begin(); it != featureList->end(); it++) {
            if (clustering[*it] == UNCLASSIFIED) {
                // Try to classify point
	      this->start_point = it;
                if (expandCluster(*it, cluster_id)) {
                    // New cluster found.
                    cluster_id++;
                    cluster_count++;
                }
            }
        }
        return clustering;
    }

    template < typename T>
    void DBScan<T>::reset()
    {
        clustering.clear();
        number_of_points = 0;
        cluster_count = 0;
        cluster_id = FIRST_CLUSTER_ID;
    }

    // All points in point cloud are being initialized as unclassified.
    template < typename T>
    void DBScan<T>::initialize()
    {
        if(!clustering.empty()) {
            reset();
        }

        cluster_id = FIRST_CLUSTER_ID;
        number_of_points = featureList->size();

        for(typename std::list<T*>::iterator it = featureList->begin(); it != featureList->end();  it++) {
            classify(*it, UNCLASSIFIED);
        }
    }

    template < typename T>
    bool DBScan<T>::expandCluster(T *start_point, int cluster_id)
    {
        // Neighbor points in epsilon radius
        std::vector<T*> seeds = neighbors(start_point);
        if (seeds.size() < min_pts) {
            // start_point has not enough neighbors in epsilon radius (is no "Kernobjekt").
            // Therefore it does not belong to any cluster.
            clustering[start_point] = NOISE;
            return false;
        }

        // New cluster found. The start point and all its neighbors belong to it. Assign cluster ID.
        clustering[start_point] = cluster_id;
        for(typename std::vector<T*>::iterator it = seeds.begin(); it != seeds.end(); it++) {
            clustering[*it] = cluster_id;
        }

        while(!seeds.empty()) {
            const int PICK_POS = 0; // TODO maybe improve choice using heuristic
            T* seed = seeds[PICK_POS];
            typename std::vector<T*> currentSeeds = neighbors(seed);

            if(currentSeeds.size() >= min_pts || speedup) {
                // Current Point has enough neighbors in epsilon radius.
                BOOST_FOREACH( T* currentSeed, currentSeeds ) {
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

    template < typename T>
    void DBScan<T>::classify(T *point, int cluster_id)
    {
        bool ret = clustering.insert(std::pair<T*, int>(point, cluster_id)).second;
        //std::cout << (ret ? "Inserted point " : "Point already inserted: ") << machine_learning::pointToString(point) << std::endl;
    }

    template < typename T>
    std::vector<T*> DBScan<T>::neighbors(T *point) {
        typename std::list<T*>::iterator it;
        typename std::vector<T*> neighbors;
        for(it = featureList->begin(); it != featureList->end();it++) {
            if(*it == point) {
                // Do not count yourself!
                continue;
            }

            /* Adjust epsilon value if desired */
            double adjusted_epsilon = epsilon;

            if(use_dynamic_epsilon) {
                // Adjust epsilon depending on the distance between point and origin and the dynamic epsilon weight.
                adjusted_epsilon = epsilon * point->norm() * dynamic_epsilon_weight;
            }

            // Check for neighbor
            if(pDistance(point, *it) <= adjusted_epsilon) {
                neighbors.push_back(*it);
		
		if(speedup && it != this->start_point){
		  it = featureList->erase(it);
		  
		  if(it != featureList->begin()){
		    it--;
		  }
		  
		}
		
            }
	      
        }
        return neighbors;
    }

    template < typename T>
    double DBScan<T>::euclidean_distance(T *p1, T *p2)
    {
      return ( *p1 - *p2).norm();
    }

    template < typename T>
    int DBScan<T>::getClusterCount()
    {
        return cluster_count;
    }

    template < typename T>
    int DBScan<T>::getNoiseCount()
    {
        int nc = 0;
        for(typename std::map<T*, int>::iterator it = clustering.begin(); it != clustering.end(); it++) {
            if(it->second == NOISE) {
                nc++;
            }
        }
        return nc;
    }    

} // end namespace machine_learning

#endif // _MACHINE_LEARNING_DBSCAN_HPP_
