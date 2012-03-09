/* ----------------------------------------------------------------------------
 * random_numbers.h
 * written by Christoph Mueller, Oct 2011
 * University of Bremen
 * ----------------------------------------------------------------------------
*/

#ifndef _MACHINE_LEARNING_RANDOM_NUMBERS_HPP_
#define _MACHINE_LEARNING_RANDOM_NUMBERS_HPP_

#include <boost/random/linear_congruential.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>

namespace machine_learning {

typedef boost::variate_generator<boost::minstd_rand&, boost::uniform_real<> > 
    UniformRandom;
typedef boost::variate_generator<boost::minstd_rand&, boost::normal_distribution<> > 
    NormalRandom;

class Random {
  public:
      static UniformRandom uniform(double min, double max);

      static NormalRandom gaussian(double mean, double variance);

  protected:
      static boost::minstd_rand& seed();

  private:
      Random() {}
      Random(const Random& random) {}
      const Random& operator=(const Random& random) { return random; }
};

}

#endif
