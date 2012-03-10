#include <machine_learning/NeuralNetwork.hpp>
#include <stdlib.h>

using namespace machine_learning;

int main(int argc, char** argv)
{
    NeuralLayer* input = new NeuralLayer(7);
    NeuralLayer* output = new NeuralLayer(1, 1.0, NeuralLayer::LINEAR);
    input->connect_to(output);

    NeuralNetwork nn(1, 1.0, input, output);

    delete input;
    delete output;

    return EXIT_SUCCESS;
}
