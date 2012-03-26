#include <machine_learning/RandomNumbers.hpp>

#include <iostream>

using namespace machine_learning;


int main(int argc, char** argv)
{
    Eigen::Matrix2d sigma;
    Eigen::Vector2d mu(0.0, 0.0);

    sigma << 1.0, 0.99, 0.99, 1.0;

    std::cerr << sigma << std::endl;

    MultiNormalRandom<2> multi = Random::multi_gaussian(mu, sigma);

    for(unsigned i = 0; i < 1000; i++) {
        Eigen::Vector2d r = multi();
        std::cout << r.x() << "\t" << r.y() << std::endl;
    }
}
