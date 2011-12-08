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
    /* This class implements the DBScan density based clustering algorithm for the specific use case of clustering obstacle points generated from sonar scan samples. For further information consult the following book:
     * Ester, M.; Sander, J.: Knowledge Discovery in Databases - Techniken und Anwendungen. 2000. Springer. Berlin.
     */
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
		 * @param min_pts Minimum point count needed in epsilon radius of a point p to define p as
		 *                core point.
		 * @param epsilon Radius from point p in which's range have to be min_pts points to define
		 *                p as core point.
		 * @param use_z   This flag defines whether to interpret the vector as 2D or 3D. If use_z is false, only the
		 *                X and Y coordinates are concerned. This is equal to (X,Y,0).
		 */
		DBScan(std::list<base::Vector3d*>* featureList, unsigned int min_pts, double epsilon, bool use_z = false);

		/* Scans the pointcloud for clusters.
		 */
		std::map<base::Vector3d*, int> scan();
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
		void reset();
		bool expandCluster(base::Vector3d *start_point, int cluster_id);
		void classify(base::Vector3d *point, int cluster_id);
		std::vector<base::Vector3d*> neighbors(base::Vector3d *point);
    };

} // end namespace machine_learning

#endif // _MACHINE_LEARNING_DBSCAN_HPP_
