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

    base::Point p;
    p[0] = 1;
    p[1] = 2;
    p[2] = 3;
    pc.points.push_back(p);
    p[0] = 6;
    p[1] = 7;
    p[2] = 8;
    pc.points.push_back(p);

    std::map<base::Point*, int>::iterator it;
    std::map<base::Point*, int> clustering = dbscan.scan(&pc, 4, 2.0);

    int count = 0;
    std::cout << "Clustering:" << std::endl;
    for(it = clustering.begin(); it != clustering.end(); it++) {
        std::cout << machine_learning::pointToString(it->first) << " clustered as " << it->second << std::endl;
    }

	return 0;
}
