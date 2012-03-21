#ifndef _MACHINE_LEARNING_NEURAL_NETWORKS_NNHPP_
#define _MACHINE_LEARNING_NEURAL_NETWORKS_NNHPP_

#include <Eigen/Core>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include "RandomNumbers.hpp"
#include "Data.hpp"

namespace machine_learning {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ParamMatrix;

class NeuralLayer;

class NeuralNetwork {
 public:
     NeuralNetwork(unsigned inputs, NeuralLayer* input, NeuralLayer* output);
     ~NeuralNetwork();

     const Vector& forward_propagation(const Vector& input);
     void back_propagation(const Vector& x, const Vector& y, double alpha);

     void reset_parameters(double variance);

     double theta_sum();

     class iterator {
       public:
          iterator(bool reverse, NeuralLayer* layer);

          iterator& operator++();
          NeuralLayer* operator*();

          bool has_layer();

       private:
          std::queue<NeuralLayer*> next_layer;
          std::set<NeuralLayer*> visit_layer;
          bool reverse;
     };

     iterator breadth_first();
     iterator breadth_first_reverse();

 private:
     NeuralLayer* input_layer;
     NeuralLayer* output_layer;
     unsigned inputs;
};


class NeuralLayer {
    friend class NeuralNetwork;

    typedef double (*Activation)(double x, double scale);
    typedef double (*Derivative)(double x, double scale);

 public:
    enum Type {
        TANH, SIGMOID, LINEAR
    };

    enum Properties {
        USE_NONE = 0,
        USE_BIAS = 1, 
        USE_CACHING = 2
    };

    NeuralLayer(unsigned nodes, double scale = 1.0, Type type = TANH, unsigned flags = USE_NONE);
    ~NeuralLayer();

    void  connect_to(NeuralLayer* layer);

    const Vector& last_computation();
    Vector input_vector(const Vector& inputs);
    Vector input_vector();

    void reset_output_vector(const Vector& vector);

    ParamMatrix& parameters() { return theta; }

    const unsigned NODES;
    const bool BIAS;
    const bool CACHING;
    const double SCALE;

 private:
    Activation activation;
    Derivative derivative;

    ParamMatrix theta;
    Vector computation;
    Vector error;
    Vector cache;

    unsigned input_dim;
    
    std::vector<NeuralLayer*> prev;
    std::vector<NeuralLayer*> next;

 private:
    static double _sigmoid(double x, double scale);
    static double _sigmoid_derivative(double x, double scale);
    static double _tanh(double x, double scale);
    static double _tanh_derivative(double x, double scale);
    static double _linear(double x, double scale);
    static double _linear_derivative(double x, double scale);

    Vector derivative_cmpwise(const Vector& v) const;
    Vector activation_cmpwise(const Vector& v) const;
};


} // namespace machine_learning

#endif
