#include "RandomNumbers.hpp"
#include <boost/cstdint.hpp>
#include <time.h>

namespace machine_learning {

boost::minstd_rand& Random::seed() {
    static boost::minstd_rand instance(static_cast<uint32_t>(time(0)));

    return instance;
}


UniformRealRandom Random::uniform_real(double min, double max) {    
    boost::uniform_real<> dist(min, max);
    return UniformRealRandom(seed(), dist);
}

UniformIntRandom Random::uniform_int(int min, int max) {    
    boost::uniform_int<> dist(min, max);
    return UniformIntRandom(seed(), dist);
}


NormalRandom Random::gaussian(double mean, double variance) {
    boost::normal_distribution<> dist(mean, variance);
    return NormalRandom(seed(), dist);
}

} // namespace machine_learning

