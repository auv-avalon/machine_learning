#include "RandomNumbers.hpp"
#include <boost/cstdint.hpp>
#include <time.h>

namespace machine_learning {

boost::minstd_rand& Random::seed() {
    static boost::minstd_rand instance(static_cast<uint32_t>(time(0)));

    return instance;
}

UniformRandom Random::uniform(double min, double max) {    
    boost::uniform_real<> dist(min, max);
    return UniformRandom(seed(), dist);
}

NormalRandom Random::gaussian(double mean, double variance) {
    boost::normal_distribution<> dist(mean, variance);
    return NormalRandom(seed(), dist);
}

} // namespace eras

