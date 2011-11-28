#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DBScan_test

#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include "../src/DBScan.hpp"

using namespace machine_learning;

BOOST_AUTO_TEST_CASE(euclidean_distance_test) {
    base::Vector3d p1(1,2,3);
    base::Vector3d p2(1,2,3);
    base::Vector3d p3(-2,4,5);

    // Equal points => distance = 0
    BOOST_CHECK_EQUAL(DBScan::euclidean_distance(p1, p2, true), 0.0);

    // Commutative points
    BOOST_CHECK_EQUAL(DBScan::euclidean_distance(p2, p3, true), DBScan::euclidean_distance(p3, p2, true));

    // Zero
    base::Vector3d p_zero_1(0,0,0);
    base::Vector3d p_zero_2(0,0,0);
    BOOST_CHECK_EQUAL(DBScan::euclidean_distance(p_zero_1, p_zero_2, true), 0.0);

    // Ignore Z Dimension
    base::Vector3d p4(3,4,5);
    base::Vector3d p4_no_z(3,4,0);
    base::Vector3d p5(-2,8,-3);
    base::Vector3d p5_no_z(-2,8,0);
    BOOST_CHECK_EQUAL(DBScan::euclidean_distance(p4, p5, false), DBScan::euclidean_distance(p4_no_z, p5_no_z, false));

    /* Spot checks */
    base::Vector3d p6(5,5,0);
    base::Vector3d p7(2,3,0);
    BOOST_CHECK_CLOSE(DBScan::euclidean_distance(p6, p7, true), 3.60555, 0.001);

    base::Vector3d p8(5,5,-2);
    base::Vector3d p9(-2,3,-3);
    BOOST_CHECK_CLOSE(DBScan::euclidean_distance(p8, p9, true), 7.34846, 0.001);

    base::Vector3d p10(4,1,2);
    base::Vector3d p11(1,1,2);
    BOOST_CHECK_EQUAL(DBScan::euclidean_distance(p10, p11, true), 3.0);
}


}

// EOF
