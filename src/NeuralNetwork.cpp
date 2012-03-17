#include "NeuralNetwork.hpp"

#include <stdexcept>
#include <math.h>

namespace machine_learning {

NeuralNetwork::NeuralNetwork(unsigned inputs, double variance, NeuralLayer* input, NeuralLayer* output)
    : initializer(Random::gaussian(0.0, variance))
{
    this->input_layer = input;
    this->output_layer = output;
    this->inputs = inputs;

    initializeParameters(output);
}


NeuralNetwork::~NeuralNetwork()
{
    // TODO: Release all layers
}


const Vector& NeuralNetwork::forward_propagation(const Vector& input)
{
    std::queue<NeuralLayer*> next_layer;
    std::set<NeuralLayer*> visit_layer;

    next_layer.push(input_layer);

    while( !next_layer.empty() ) {
        NeuralLayer* layer = next_layer.front();
        next_layer.pop();
        visit_layer.insert(layer);

        std::vector<NeuralLayer*>::const_iterator it;

        for(it = layer->next.begin(); it != layer->next.end(); it++) {
            if(visit_layer.find(*it) == visit_layer.end()) {
                next_layer.push(*it);
            }                
        }

        Vector y = (layer == input_layer) 
            ? layer->theta.transpose() * layer->input_vector()
            : layer->theta.transpose() * layer->input_vector(input);

        for(unsigned i = 0; i < y.rows(); i++) {
            layer->computation(i) = layer->activation(y(i), layer->SCALE);
        }
    }

    return output_layer->computation;
}


void NeuralNetwork::back_propagation(const Vector& x, const Vector& y, double alpha)
{
    const Vector& value = forward_propagation(x);
}


double NeuralNetwork::theta_sum() const
{
    std::queue<NeuralLayer*> next_layer;
    std::set<NeuralLayer*> visit_layer;

    next_layer.push(input_layer);
    double sum = 0;

    while( !next_layer.empty() ) {
        std::vector<NeuralLayer*>::iterator it;

        NeuralLayer* layer = next_layer.front();

        sum += layer->theta.sum();
            
        visit_layer.insert(layer);
        next_layer.pop();

        for(it = layer->next.begin(); it != layer->next.end(); it++) {
            if(visit_layer.find(*it) == visit_layer.end())
                next_layer.push(*it);
        }
    }

    return sum;
}


void NeuralNetwork::initializeParameters(NeuralLayer* output)
{
    std::queue<NeuralLayer*> next_layer;
    std::set<NeuralLayer*> visit_layer;

    next_layer.push(output_layer);

    while( !next_layer.empty() ) {
        std::vector<NeuralLayer*>::iterator it;

        NeuralLayer* layer = next_layer.front();

        visit_layer.insert(layer);
        next_layer.pop();

        for(it = layer->prev.begin(); it != layer->prev.end(); it++) {
            if(visit_layer.find(*it) == visit_layer.end())
                next_layer.push(*it);
        }

        layer->theta.resize(layer->input_dim, layer->NODES);

        for(unsigned i = 0; i < layer->input_dim; i++)
            for(unsigned j = 0; j < layer->NODES; j++)
                layer->theta(i,j) = initializer();
    }
}


NeuralLayer::NeuralLayer(unsigned nodes, double scale, Type type, bool bias)
    : NODES(nodes), BIAS(bias), SCALE(scale), input_dim(0)
{
    switch(type) {
        case TANH:
            activation = &NeuralLayer::_tanh;
            derivative = &NeuralLayer::_tanh_derivative;
            break;
        case SIGMOID:
            activation = &NeuralLayer::_sigmoid;
            derivative = &NeuralLayer::_sigmoid_derivative;
            break;
        case LINEAR:
            activation = &NeuralLayer::_linear;
            derivative = &NeuralLayer::_linear_derivative;
            break;
        default:
            throw std::runtime_error("unknown activation functions for this layer");
    }
}


NeuralLayer::~NeuralLayer()
{
}

const Vector& NeuralLayer::last_computation() 
{
    return computation;
}


Vector NeuralLayer::input_vector(const Vector& inputs)
{
    std::vector<NeuralLayer*>::iterator it;

    Vector input(input_dim + inputs.rows(), 1);

    unsigned index = 0;

    for(unsigned i = 0; i < inputs.rows(); i++)
        input(index++) =  inputs(i);
    
    for(it = prev.begin(); it != prev.end(); it++) {
        Vector v = (*it)->last_computation();

        for(unsigned i = 0; i < v.rows(); i++) 
            input(index++) = v(i);
    }

    return input;
}


Vector NeuralLayer::input_vector()
{
    std::vector<NeuralLayer*>::iterator it;

    Vector input(input_dim, 1);

    unsigned index = 0;
    
    for(it = prev.begin(); it != prev.end(); it++) {
        Vector v = (*it)->last_computation();

        for(unsigned i = 0; i < v.rows(); i++) 
            input(index++) = v(i);
    }

    return input;
}


Vector NeuralLayer::error_vector(const Vector& y)
{
    return computation - y;
}


void NeuralLayer::reset_output_vector(const Vector& vector)
{
    computation = vector;
}


void NeuralLayer::connect_to(NeuralLayer* layer)
{
    layer->prev.push_back(this);
    this->next.push_back(layer);
    this->input_dim += layer->NODES;
}


double NeuralLayer::_sigmoid(double x, double scale)
{
    return 1.0 / (1.0 + exp(-scale * x));
}


double NeuralLayer::_sigmoid_derivative(double x, double scale)
{
    return scale * _sigmoid(x, scale) * (1.0 - _sigmoid(x, scale));
}


double NeuralLayer::_tanh(double x, double scale)
{
    return tanh(scale * x);
}


double NeuralLayer::_tanh_derivative(double x, double scale)
{
    return scale * (1.0 - tanh(x * scale) * tanh(x * scale));
}


double NeuralLayer::_linear(double x, double scale)
{
    return scale * x;
}


double NeuralLayer::_linear_derivative(double x, double scale)
{
    return scale;
}


}
