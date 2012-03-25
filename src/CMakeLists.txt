set(CPP
    DBScan.cpp
    RandomNumbers.cpp
    NeuralNetwork.cpp
    Data.cpp
    )

set(HPP
    DBScan.hpp
    ClusteringUtils.hpp
    GaussianParameters.hpp
    RandomNumbers.hpp
    NeuralNetwork.hpp
    Data.hpp
    )

rock_library(machine_learning
    SOURCES ${CPP}
    HEADERS ${HPP}
    DEPS_PKGCONFIG base-types eigen3)

rock_executable(machine_learning_bin Main.cpp
    DEPS machine_learning
    DEPS_PKGCONFIG base-types eigen3)

