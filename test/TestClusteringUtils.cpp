#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ClusteringUtils_test

#include "../src/ClusteringUtils.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>

using namespace machine_learning;

BOOST_AUTO_TEST_CASE(getCentroid_test) {
    base::Vector3d p1(2,2,2);
    base::Vector3d p2(4,4,4);
    
    std::vector<base::Vector3d*> vec;
    vec.push_back(&p1);
    vec.push_back(&p2);
    
    base::Vector3d result = centroid(vec);
    BOOST_CHECK_EQUAL(base::Vector3d(3,3,3), centroid(vec));
}
