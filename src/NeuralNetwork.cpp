#include "NeuralNetwork.hpp"

#include <stdexcept>
#include <math.h>
#include <queue>
#include <set>

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

        Vector x(layer->theta.rows());
        unsigned index = 0;
        for(unsigned i = 0; i < input.rows(); i++) {
            x(index) = input(i);
            index++;
        }


        std::vector<NeuralLayer*>::iterator it;
        for(it = layer->prev.begin(); it != layer->prev.end(); it++) { 
            for(unsigned i = 0; i < (*it)->last_computation.rows(); i++) {
                x(index) = (*it)->last_computation(i);
                index++;
            } 
        }

        for(it = layer->next.begin(); it != layer->next.end(); it++) {
            if(visit_layer.find(*it) == visit_layer.end()) {
                next_layer.push(*it);
            }                
        }

        Vector y = layer->theta.transpose() * x;

        for(unsigned i = 0; i < y.rows(); i++) {
            layer->last_computation(i) = layer->activation(y(i), layer->SCALE);
        }
    }

    return output_layer->last_computation;
}


void NeuralNetwork::initializeParameters(NeuralLayer* output)
{
    std::queue<NeuralLayer*> next;
    std::set<NeuralLayer*> visit;

    next.push(output_layer);

    while( !next.empty() ) {
        std::vector<NeuralLayer*>::iterator it;

        NeuralLayer* layer = next.front();

        visit.insert(layer);
        next.pop();

        unsigned input_nodes = 0;
        for(it = layer->prev.begin(); it != layer->prev.end(); it++) {
            input_nodes += (*it)->NODES;

            if(visit.find(*it) == visit.end())
                next.push(*it);
        }

        if(layer == input_layer)
            input_nodes += inputs;
        
        layer->theta.resize(input_nodes, layer->NODES);

        for(unsigned i = 0; i < input_nodes; i++)
            for(unsigned j = 0; j < layer->NODES; j++)
                layer->theta(i,j) = initializer();
    }
}


NeuralLayer::NeuralLayer(unsigned nodes, double scale, Type type, bool bias)
    : NODES(nodes), BIAS(bias), SCALE(scale)
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


void NeuralLayer::connect_to(NeuralLayer* layer)
{
    layer->prev.push_back(this);
    this->next.push_back(layer);
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
