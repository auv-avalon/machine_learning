#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <cstdlib> // rand()
#include <machine_learning/DBScan.hpp>
#include <sonar_detectors/SonarDetectorTypes.hpp>

std::list<sonar_detectors::obstaclePoint>* pointCloud;

std::list<sonar_detectors::obstaclePoint>* generateRandomPointCloud(int size) {
    delete pointCloud;
    pointCloud = new std::list<sonar_detectors::obstaclePoint>();
    int coordRange = 40;
    for(int i = 0; i < size; i++) {
        sonar_detectors::obstaclePoint p;
	p.position[0] = rand() % coordRange;
        p.position[1] = rand() % coordRange;
        p.position[2] = rand() % coordRange;
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

    std::list<sonar_detectors::obstaclePoint>* pl = generateRandomPointCloud(50);
    machine_learning::DBScan dbscan(pl, 3, 4.0); // ignoring depth! use_z = 0

    /*
    double dist = dbscan.euclidean_distance(p1, p2, false);
    std::cout << "Distance between p1 and p2: " << dist << std::endl;
    */

    std::map<sonar_detectors::obstaclePoint*, int>::iterator it;
    std::map<sonar_detectors::obstaclePoint*, int> clustering = dbscan.scan();

    std::cout << "Clustering:" << std::endl;
    for(it = clustering.begin(); it != clustering.end(); it++) {
        std::cout << machine_learning::pointToString(*(*it).first) << " clustered as " << (*it).second << std::endl;
    }

	return 0;
}
