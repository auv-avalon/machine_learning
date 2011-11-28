#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <cstdlib> // rand()
#include <machine_learning/DBScan.hpp>

std::list<base::Vector3d>* pointCloud;

std::list<base::Vector3d>* generateRandomPointCloud(int size) {
    delete pointCloud;
    pointCloud = new std::list<base::Vector3d>();
    int coordRange = 40;
    for(int i = 0; i < size; i++) {
        base::Vector3d p;
        p[0] = rand() % coordRange;
        p[1] = rand() % coordRange;
        p[2] = rand() % coordRange;
        pointCloud->push_back(p);
    }
    return pointCloud;
}

int main(int argc, char** argv)
{
    std::cout << "Welcome to the testing of machine_learning" << std::endl;

    //int seed = 1320319343;
    int seed = time(NULL);
    std::cout << seed << std::endl;
    srand(seed);

    std::list<base::Vector3d>* pl = generateRandomPointCloud(50);
    machine_learning::DBScan dbscan(pl, 3, 4.0); // ignoring depth! use_z = 0

    /*
    double dist = dbscan.euclidean_distance(p1, p2, false);
    std::cout << "Distance between p1 and p2: " << dist << std::endl;
    */

    std::map<base::Vector3d*, int>::iterator it;
    std::map<base::Vector3d*, int> clustering = dbscan.scan();

    std::cout << "Clustering:" << std::endl;
    for(it = clustering.begin(); it != clustering.end(); it++) {
        std::cout << machine_learning::pointToString(*(*it).first) << " clustered as " << (*it).second << std::endl;
    }

	return 0;
}
