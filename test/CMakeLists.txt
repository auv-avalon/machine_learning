rock_testsuite(dbscan_test DEPS machine_learning DEPS_PKGCONFIG sonar_detectors base-types eigen3 SOURCES Test.cpp)

rock_testsuite(machine_learning_test DEPS machine_learning DEPS_PKGCONFIG sonar_detectors base-types eigen3 SOURCES TestClusteringUtils.cpp)
