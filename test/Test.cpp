#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE DBScan_test

#include <iostream>
#include <stdio.h>
#include "../src/DBScan.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>

using namespace machine_learning;

BOOST_AUTO_TEST_CASE(euclidean_distance_test) {
    base::Vector3d p1(1,2,3);
    base::Vector3d p2(1,2,3);
    base::Vector3d p3(-2,4,5);

    // Equal points => distance = 0
    BOOST_CHECK_EQUAL(DBScan<base::Vector3d>::euclidean_distance(&p1, &p2), 0.0);

    // Commutative points
    BOOST_CHECK_EQUAL(DBScan<base::Vector3d>::euclidean_distance(&p2, &p3), DBScan<base::Vector3d>::euclidean_distance(&p3, &p2));

    // Zero
    base::Vector3d p_zero_1(0,0,0);
    base::Vector3d p_zero_2(0,0,0);
    BOOST_CHECK_EQUAL(DBScan<base::Vector3d>::euclidean_distance(&p_zero_1, &p_zero_2), 0.0);

    /* Spot checks */
    base::Vector3d p6(5,5,0);
    base::Vector3d p7(2,3,0);
    BOOST_CHECK_CLOSE(DBScan<base::Vector3d>::euclidean_distance(&p6, &p7), 3.60555, 0.001);

    base::Vector3d p8(5,5,-2);
    base::Vector3d p9(-2,3,-3);
    BOOST_CHECK_CLOSE(DBScan<base::Vector3d>::euclidean_distance(&p8, &p9), 7.34846, 0.001);

    base::Vector3d p10(4,1,2);
    base::Vector3d p11(1,1,2);
    BOOST_CHECK_EQUAL(DBScan<base::Vector3d>::euclidean_distance(&p10, &p11), 3.0);
}

BOOST_AUTO_TEST_CASE(clustering_test_1) {
    /* Point naming scheme: p_<cluster_id>_<point_number> */

    /* Cluster 0 */
    base::Vector3d p_0_0(1.5, 8,   0);
    base::Vector3d p_0_1(2,   6,   0);
    base::Vector3d p_0_2(1.0, 7.5, 0);

    /* Noise */
    base::Vector3d p_n_0(1.5, 2, 0); // far away from c_0
    base::Vector3d p_n_1(-5,  9, 0); // very far away from c_0
    base::Vector3d p_n_2(5.5, 6, 0); // almost in range of p_0_1

    std::list<base::Vector3d*> featureList;
    featureList.push_back(&p_0_0);
    featureList.push_back(&p_0_1);
    featureList.push_back(&p_0_2);
    featureList.push_back(&p_n_0);
    featureList.push_back(&p_n_1);
    featureList.push_back(&p_n_2);

    DBScan<base::Vector3d> dbs(&featureList, 2, 3.0, false);
    std::map<base::Vector3d*, int> clustering = dbs.scan();

    BOOST_CHECK_EQUAL(clustering.size(), featureList.size());
    BOOST_CHECK_EQUAL(dbs.getClusterCount(), 1);
    BOOST_CHECK_EQUAL(dbs.getNoiseCount(), 3);

    BOOST_CHECK_EQUAL(clustering[&p_0_0],  0);
    BOOST_CHECK_EQUAL(clustering[&p_0_1],  0);
    BOOST_CHECK_EQUAL(clustering[&p_0_2],  0);
    BOOST_CHECK_EQUAL(clustering[&p_n_0], -2);
    BOOST_CHECK_EQUAL(clustering[&p_n_1], -2);
    BOOST_CHECK_EQUAL(clustering[&p_n_2], -2);
}

BOOST_AUTO_TEST_CASE(clustering_test_2) {

    base::Vector3d p_a(-2.5, -2.5, 0);
    base::Vector3d p_b(-1.5, -1.0, 0);
    base::Vector3d p_c(-0.3, -0.2, 0);
    base::Vector3d p_d( 1.7,  1.5, 0);
    base::Vector3d p_e( 2.0,  2.5, 0);
    base::Vector3d p_f( 0.7, -0.5, 0);
    base::Vector3d p_g(-0.9,  0.9, 0);

    std::list<base::Vector3d*> featureList;
    featureList.push_back(&p_a);
    featureList.push_back(&p_b);
    featureList.push_back(&p_c);
    featureList.push_back(&p_d);
    featureList.push_back(&p_e);
    featureList.push_back(&p_f);
    featureList.push_back(&p_g);

    DBScan<base::Vector3d> dbs(&featureList, 3, 3.0, false);
    std::map<base::Vector3d*, int> clustering = dbs.scan();

    BOOST_CHECK_EQUAL(clustering.size(), featureList.size());
    BOOST_CHECK_EQUAL(dbs.getClusterCount(), 1);
    BOOST_CHECK_EQUAL(dbs.getNoiseCount(), 0);

    BOOST_CHECK_EQUAL(clustering[&p_a], 0);
    BOOST_CHECK_EQUAL(clustering[&p_b], 0);
    BOOST_CHECK_EQUAL(clustering[&p_c], 0);
    BOOST_CHECK_EQUAL(clustering[&p_d], 0);
    BOOST_CHECK_EQUAL(clustering[&p_e], 0);
    BOOST_CHECK_EQUAL(clustering[&p_f], 0);
    BOOST_CHECK_EQUAL(clustering[&p_g], 0);
}

BOOST_AUTO_TEST_CASE(clustering_test_3) {

    base::Vector3d p_a(-2.5, 0, 0);
    base::Vector3d p_b(-1.5, 0, 0);

    std::list<base::Vector3d*> featureList;
    featureList.push_back(&p_a);
    featureList.push_back(&p_b);

    DBScan<base::Vector3d> dbs(&featureList, 1, 1.0, false);
    std::map<base::Vector3d*, int> clustering = dbs.scan();

    BOOST_CHECK_EQUAL(clustering.size(), featureList.size());
    BOOST_CHECK_EQUAL(dbs.getClusterCount(), 1);
    BOOST_CHECK_EQUAL(dbs.getNoiseCount(), 0);

    BOOST_CHECK_EQUAL(clustering[&p_a], 0);
    BOOST_CHECK_EQUAL(clustering[&p_b], 0);
}
/*
BOOST_AUTO_TEST_CASE(clustering_test_only_noise) {

    base::Vector3d p_a(-2.5, -2.5, 0);
    base::Vector3d p_b( 4.5,  6.0, 0);
    base::Vector3d p_c( 8.0, -12.5, 0);

    std::list<base::Vector3d*> featureList;
    featureList.push_back(&p_a);
    featureList.push_back(&p_b);
    featureList.push_back(&p_c);

    DBScan dbs(&featureList, 3, 3.0, false);
    std::map<base::Vector3d*, int> clustering = dbs.scan();

    BOOST_CHECK_EQUAL(clustering.size(), featureList.size());
    BOOST_CHECK_EQUAL(dbs.getClusterCount(), 0);
    BOOST_CHECK_EQUAL(dbs.getNoiseCount(), 3);

    BOOST_CHECK_EQUAL(clustering[&p_a], -2);
    BOOST_CHECK_EQUAL(clustering[&p_b], -2);
    BOOST_CHECK_EQUAL(clustering[&p_c], -2);
}

BOOST_AUTO_TEST_CASE(clustering_test_empty_pointcloud) {

    std::list<base::Vector3d*> featureList;
    BOOST_CHECK(featureList.empty());

    DBScan dbs(&featureList, 3, 3.0, false);
    std::map<base::Vector3d*, int> clustering = dbs.scan();

    BOOST_CHECK_EQUAL(clustering.size(), featureList.size());
    BOOST_CHECK_EQUAL(dbs.getClusterCount(), 0);
    BOOST_CHECK_EQUAL(dbs.getNoiseCount(), 0);
}
*/
// EOF
