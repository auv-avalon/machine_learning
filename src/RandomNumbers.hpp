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
#include "GaussianParameters.hpp"

namespace machine_learning {

typedef boost::variate_generator<boost::minstd_rand&, boost::uniform_real<> > 
    UniformRandom;
typedef boost::variate_generator<boost::minstd_rand&, boost::normal_distribution<> > 
    NormalRandom;


template <int DIM>
class MultiNormalRandom {
      MultiNormalRandom(boost::minstd_rand& seed, const VECTOR_XD(DIM)& mean, const MATRIX_XD(DIM)& cov)
          : NormalRandom(seed, boost::normal_distribution<>(0.0, 1.0)), parameters(mean, cov)
      {}

      VECTOR_XD(DIM) operator()() {
          VECTOR_XD(DIM) random_number;

          if(!parameters.covariance.ldlt().isPositive())
              throw new std::runtime_error("Covariance is not semi-positive definit");

          for(unsigned i = 0; i < DIM; i++)
              random_number(i) = Normal();

          return random_number * parameters.covariance.ldlt().matrixLDLT() + parameters.mean;
      }

  private:
      NormalRandom Normal;
      GaussParam<DIM> parameters;
};


class Random {
  public:
      static UniformRandom uniform(double min, double max);

      static NormalRandom gaussian(double mean, double variance);

      template <int DIM>
      static MultiNormalRandom<DIM> multi_gaussian(const VECTOR_XD(DIM)& mean, const MATRIX_XD(DIM)& cov) {
          return MultiNormalRandom<DIM>(seed(), mean, cov);
      }

      template <int DIM>
      static MultiNormalRandom<DIM> multi_gaussian(const GaussParam<DIM>& parameters) {
          return MultiNormalRandom<DIM>(seed(), parameters.mean, parameters.covariance);
      }



  protected:
      static boost::minstd_rand& seed();

  private:
      Random() {}
      Random(const Random& random) {}
      const Random& operator=(const Random& random) { return random; }
};


}

#endif
