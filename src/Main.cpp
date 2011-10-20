#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <machine_learning/DBScan.hpp>
#include <base/samples/pointcloud.h>

int main(int argc, char** argv)
{
    base::samples::Pointcloud pc;
	machine_learning::DBScan dbscan;

    base::Point p1;
    p1[0] = 1;
    p1[1] = 2;
    p1[2] = 3;
    pc.points.push_back(p1);

    base::Point p2;
    p2[0] = 6;
    p2[1] = 7;
    p2[2] = 8;
    pc.points.push_back(p2);

    double dist = dbscan.euclidean_distance(&p1, &p2, false);
    std::cout << "Distance between p1 and p2: " << dist << std::endl;

    std::map<base::Point*, int>::iterator it;
    std::map<base::Point*, int> clustering = dbscan.scan(&pc, 4, 2.0);

    std::cout << "Clustering:" << std::endl;
    for(it = clustering.begin(); it != clustering.end(); it++) {
        std::cout << machine_learning::pointToString(it->first) << " clustered as " << it->second << std::endl;
    }

	return 0;
}
