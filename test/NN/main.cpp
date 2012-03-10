#include <machine_learning/NeuralNetwork.hpp>
#include <machine_learning/Data.hpp>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace machine_learning;

int main(int argc, char** argv)
{
    NeuralLayer* input = new NeuralLayer(7);
    NeuralLayer* output = new NeuralLayer(1, 1.0, NeuralLayer::LINEAR);
    input->connect_to(output);

    NeuralNetwork nn(1, 1.0, input, output);


    std::fstream file("sin.data");

    Data d;

    file >> d;

    std::cout << d << std::endl;

    delete input;
    delete output;

    return EXIT_SUCCESS;
}
