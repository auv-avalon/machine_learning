rock_testsuite(dbscan_test DEPS machine_learning DEPS_PKGCONFIG base-types eigen3 SOURCES Test.cpp)

rock_testsuite(machine_learning_test DEPS machine_learning DEPS_PKGCONFIG base-types eigen3 SOURCES TestClusteringUtils.cpp)

rock_executable(sinus_learner NN/main.cpp DEPS machine_learning DEPS_PKGCONFIG base-types eigen3)

rock_executable(random_gen Random/test.cpp DEPS machine_learning DEPS_PKGCONFIG base-types eigen3)
