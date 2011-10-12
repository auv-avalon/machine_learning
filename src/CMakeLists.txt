rock_library(machine_learning
    SOURCES Dummy.cpp
    HEADERS Dummy.hpp)

rock_executable(machine_learning_bin Main.cpp
    DEPS machine_learning)

