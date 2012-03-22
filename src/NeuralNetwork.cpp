#include "NeuralNetwork.hpp"

#include <boost/assert.hpp>
#include <stdexcept>
#include <math.h>

namespace machine_learning {

NeuralNetwork::NeuralNetwork(unsigned inputs, NeuralLayer* input, NeuralLayer* output)
{
    this->input_layer = input;
    this->output_layer = output;
    this->inputs = inputs;
}


NeuralNetwork::~NeuralNetwork()
{}


NeuralNetwork::iterator::iterator(bool reverse, NeuralLayer* layer) 
  : reverse(reverse)
{
    next_layer.push(layer);
}


NeuralNetwork::iterator& NeuralNetwork::iterator::operator++()
{
    NeuralLayer* layer = next_layer.front();
    next_layer.pop();
    visit_layer.insert(layer);

    std::vector<NeuralLayer*>::iterator it;

    if(!reverse) {
        for(it = layer->next.begin(); it != layer->next.end(); it++) {
            if(visit_layer.find(*it) == visit_layer.end())
                next_layer.push(*it);
        }
    } else {
        for(it = layer->prev.begin(); it != layer->prev.end(); it++) {
            if(visit_layer.find(*it) == visit_layer.end())
                next_layer.push(*it);
        }
    }

    return *this;
}


NeuralLayer* NeuralNetwork::iterator::operator*()
{
    return next_layer.front();
}


bool NeuralNetwork::iterator::has_layer()
{
    return !next_layer.empty();
}


NeuralNetwork::iterator NeuralNetwork::breadth_first()
{
    return iterator(false, input_layer);
}


NeuralNetwork::iterator NeuralNetwork::breadth_first_reverse()
{
    return iterator(true, output_layer);
}


const Vector& NeuralNetwork::forward_propagation(const Vector& input)
{
    BOOST_ASSERT(input.rows() == inputs && input.cols() == 1);

    NeuralNetwork::iterator it = breadth_first();

    while(it.has_layer()) {
        NeuralLayer* layer = *it;

        Vector y;
        if(layer == input_layer) {
            BOOST_ASSERT(layer->theta.transpose().cols() == layer->input_vector(input).rows()); 

            y = layer->theta.transpose() * layer->input_vector(input);
        } else {
            BOOST_ASSERT(layer->theta.transpose().cols() == layer->input_vector().rows());

            y = layer->theta.transpose() * layer->input_vector();
        }

        if(layer->CACHING)
            layer->cache = layer->computation;

        layer->computation = layer->activation_cmpwise(y);

        ++it;
    }

    return output_layer->computation;
}


void NeuralNetwork::back_propagation(const Vector& x, const Vector& y, double alpha)
{
    BOOST_ASSERT(x.rows() == inputs && x.cols() == 1);

    const Vector& value = forward_propagation(x);

    BOOST_ASSERT(value.rows() == y.rows() && value.cols() == y.cols());

    NeuralNetwork::iterator it = breadth_first_reverse();

    while(it.has_layer()) {
        NeuralLayer* layer = *it;

        if(layer == output_layer) {
            layer->error = (value - y).cwiseProduct(layer->derivative_cmpwise(layer->computation));
        } else {
            // case for connection to multiple successor layers
            if(layer->next.size() > 1) {
                // TODO:
                throw std::runtime_error("multiple successor layers currently not supported");
            } else {
                NeuralLayer* succ = layer->next.front();

                BOOST_ASSERT(succ->theta.cols() == succ->error.rows());

                Vector p = (succ->theta * succ->error);
                Vector reduce = succ->BIAS ? p.block(1, 0, p.rows() - 1, 1) : p;

                layer->error = (reduce).cwiseProduct(layer->derivative_cmpwise(layer->computation));
            }
        }

        ++it;
    }

    it = breadth_first();

    while(it.has_layer()) {
        NeuralLayer* layer = *it;

        Vector in = (layer == input_layer) ? layer->input_vector(x) : layer->input_vector();

        BOOST_ASSERT(in.cols() == layer->error.cols());

        layer->theta += alpha * in * -layer->error.transpose();

        if(layer->CACHING)
            layer->computation = layer->cache;

        ++it;
    }
}


double NeuralNetwork::theta_sum()
{
    double sum = 0;

    NeuralNetwork::iterator it = breadth_first();

    while(it.has_layer()) {
        NeuralLayer* layer = *it;

        sum += layer->theta.sum();

        ++it;
    }

    return sum;
}


void NeuralNetwork::reset_parameters(double variance)
{
    NormalRandom initializer = Random::gaussian(0.0, variance);

    NeuralNetwork::iterator it = breadth_first_reverse();

    while(it.has_layer()) {
        NeuralLayer* layer = *it;

        if(layer == input_layer)
            layer->input_dim += inputs;

        if(layer->BIAS)
            layer->input_dim++;

        layer->theta.resize(layer->input_dim, layer->NODES);

        for(unsigned i = 0; i < layer->input_dim; i++)
            for(unsigned j = 0; j < layer->NODES; j++)
                layer->theta(i,j) = initializer();

        ++it;
    }
}


NeuralLayer::NeuralLayer(unsigned nodes, double scale, Type type, unsigned prop)
    : NODES(nodes), 
      BIAS((prop & USE_BIAS) == USE_BIAS), 
      CACHING((prop & USE_CACHING) == USE_CACHING), 
      SCALE(scale),
      input_dim(0)
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

    Vector input(input_dim, 1);

    unsigned index = 0;

    if(BIAS) {
        input(index++) = 1.0;
    }

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

    if(BIAS) {
        input(index++) = 1.0;
    }
    
    for(it = prev.begin(); it != prev.end(); it++) {
        Vector v = (*it)->last_computation();

        for(unsigned i = 0; i < v.rows(); i++) 
            input(index++) = v(i);
    }

    return input;
}


void NeuralLayer::reset_output_vector(const Vector& vector)
{
    computation = vector;
}


void NeuralLayer::connect_to(NeuralLayer* layer)
{
    if(this->next.size() > 0) {
        throw std::runtime_error("Multiple successor layers are currently not supported");
    }

    layer->prev.push_back(this);
    layer->input_dim += this->NODES;
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


Vector NeuralLayer::derivative_cmpwise(const Vector& v) const
{
    Vector r(v.rows(), 1);
    for(unsigned index = 0; index < v.rows(); index++) {
        r(index) = this->derivative(v(index), SCALE);
    }

    return r;
}


Vector NeuralLayer::activation_cmpwise(const Vector& v) const
{
    Vector r(v.rows(), 1);
    for(unsigned index = 0; index < v.rows(); index++) {
        r(index) = this->activation(v(index), SCALE);
    }

    return r;
}




}
