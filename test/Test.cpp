#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DBScan_test

#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include "../src/DBScan.hpp"


using namespace machine_learning;

BOOST_AUTO_TEST_CASE(euclidean_distance_test) {
    //DBScan dbs();
    base::Vector3d p1(1,2,3);
    base::Vector3d p2(1,2,3);
    base::Vector3d p3(-2,4,5);

    // Equal points => distance = 0
    BOOST_CHECK_EQUAL(DBScan::euclidean_distance(p1, p2), 0.0);

    // Commutative points
    BOOST_CHECK_EQUAL(DBScan::euclidean_distance(p2, p3), DBScan::euclidean_distance(p3, p2));

    // Ignore Z Dimension
    base::Vector3d p4(3,4,5);
    base::Vector3d p4_no_z(3,4,0);
    base::Vector3d p5(-2,8,-3);
    base::Vector3d p5_no_z(-2,8,0);
    BOOST_CHECK_EQUAL(DBScan::euclidean_distance(p4, p5, false), DBScan::euclidean_distance(p4_no_z, p5_no_z, false));


    BOOST_CHECK_MESSAGE(DBScan::euclidean_distance(p1, p2) == 1.0, "This will fail."); // will fail


}

// EOF
