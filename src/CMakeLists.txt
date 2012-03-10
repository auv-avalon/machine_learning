rock_library(machine_learning
    SOURCES DBScan.cpp RandomNumbers.cpp NeuralNetwork.cpp
    HEADERS DBScan.hpp ClusteringUtils.hpp RandomNumbers.hpp NeuralNetwork.hpp)

rock_executable(machine_learning_bin Main.cpp
    DEPS machine_learning
    DEPS_PKGCONFIG base-types eigen3)

