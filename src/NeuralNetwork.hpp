#ifndef _MACHINE_LEARNING_NEURAL_NETWORKS_NNHPP_
#define _MACHINE_LEARNING_NEURAL_NETWORKS_NNHPP_

#include <Eigen/Core>
#include <vector>
#include "RandomNumbers.hpp"

namespace machine_learning {

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ParamMatrix;

class NeuralLayer;

class NeuralNetwork {
 public:
     NeuralNetwork(unsigned inputs, double variance, NeuralLayer* input, NeuralLayer* output);
     ~NeuralNetwork();

     const Vector& forward_propagation(const Vector& input);

 private:
     void initializeParameters(NeuralLayer* output);

 private:
     NeuralLayer* input_layer;
     NeuralLayer* output_layer;
     unsigned inputs;
     NormalRandom initializer;
};


class NeuralLayer {
    friend class NeuralNetwork;

    typedef double (*Activation)(double x, double scale);
    typedef double (*Derivative)(double x, double scale);

 public:
    enum Type {
        TANH, SIGMOID, LINEAR
    };

    NeuralLayer(unsigned nodes, double scale = 1.0, Type type = TANH, bool bias = false);
    ~NeuralLayer();

    void  connect_to(NeuralLayer* layer);

    const unsigned NODES;
    const bool BIAS;
    const double SCALE;

 private:
    Activation activation;
    Derivative derivative;

    ParamMatrix theta;
    Vector last_computation;
    std::vector<NeuralLayer*> prev;
    std::vector<NeuralLayer*> next;

 private:
    static double _sigmoid(double x, double scale);
    static double _sigmoid_derivative(double x, double scale);
    static double _tanh(double x, double scale);
    static double _tanh_derivative(double x, double scale);
    static double _linear(double x, double scale);
    static double _linear_derivative(double x, double scale);
};


} // namespace machine_learning

#endif
