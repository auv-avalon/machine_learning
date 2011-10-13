rock_library(machine_learning
    SOURCES DBScan.cpp
    HEADERS DBScan.hpp)

rock_executable(machine_learning_bin Main.cpp
    DEPS machine_learning)

