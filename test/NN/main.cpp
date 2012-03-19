#include <machine_learning/NeuralNetwork.hpp>
#include <machine_learning/Data.hpp>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>

using namespace machine_learning;

int main(int argc, char** argv)
{
    if(argc != 2) {
        std::cout << "Command: sinus_learner <DATAFILE>" << std::endl;
        return EXIT_FAILURE;
    }

    NeuralLayer* input = new NeuralLayer(10, 1.0, NeuralLayer::TANH, true);
    NeuralLayer* output = new NeuralLayer(1, 1.0, NeuralLayer::LINEAR, true);
    input->connect_to(output);

    NeuralNetwork nn(1, 1.0, input, output);

    std::fstream file(argv[1]);

    Data input_data(file);
    Data output_data(floor(1.0 / 0.01), 2);

    double conv = 0;

    std::cerr << std::endl;

    do {
        conv = nn.theta_sum();
        for(unsigned i = 0; i < input_data.rows(); i++) {
            Vector x = Vector::Constant(1, 1, input_data(i, 0));
            Vector t = input_data.block(i, input_data.cols() - 1, 1, 1);

            nn.back_propagation(x, t, 0.01);
        }
        fprintf(stderr, "\r Current error %.7f", fabs(conv - nn.theta_sum()));
    } while(fabs(conv - nn.theta_sum()) > 0.00001);

    std::cerr << std::endl;

    unsigned row = 0;
    for(double x = 0.0; x <= 1.0; x += 0.01) {
        Vector y = nn.forward_propagation(Vector::Constant(1, 1, x));

        output_data(row, 0) = x;
        output_data(row, 1) = y(0);

        row++;
    }


    std::cout << output_data << std::endl;


    delete input;
    delete output;

    return EXIT_SUCCESS;
}
