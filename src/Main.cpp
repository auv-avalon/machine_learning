#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <machine_learning/DBScan.hpp>
#include <sonar_detectors/SonarDetectorTypes.hpp>

int main(int argc, char** argv)
{
    std::list<sonar_detectors::obstaclePoint> pl;
	machine_learning::DBScan dbscan(&pl, 2, 2.0); // ignoring depth! use_z = 0

    sonar_detectors::obstaclePoint p1;
    p1.position[0] = -1;
    p1.position[1] = -2;
    p1.position[2] = 3;
    pl.push_back(p1);

    sonar_detectors::obstaclePoint p2;
    p2.position[0] = 1;
    p2.position[1] = 2;
    p2.position[2] = 3;
    pl.push_back(p2);

    double dist = dbscan.euclidean_distance(&p1, &p2, false);
    std::cout << "Distance between p1 and p2: " << dist << std::endl;

    std::map<sonar_detectors::obstaclePoint*, int>::iterator it;
    std::map<sonar_detectors::obstaclePoint*, int> clustering = dbscan.scan();

    std::cout << "Clustering:" << std::endl;
    for(it = clustering.begin(); it != clustering.end(); it++) {
        std::cout << machine_learning::pointToString(it->first) << " clustered as " << it->second << std::endl;
    }

	return 0;
}
