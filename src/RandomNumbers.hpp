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
#include <boost/random/uniform_int.hpp>
#include <boost/random/normal_distribution.hpp>
#include <Eigen/Cholesky>
#include "GaussianParameters.hpp"

namespace machine_learning {

typedef boost::variate_generator<boost::minstd_rand&, boost::uniform_int<> >
    UniformIntRandom;
typedef boost::variate_generator<boost::minstd_rand&, boost::uniform_real<> > 
    UniformRealRandom;
typedef boost::variate_generator<boost::minstd_rand&, boost::normal_distribution<> > 
    NormalRandom;

template <int DIM>
class MultiNormalRandom {
  public:
      MultiNormalRandom(boost::minstd_rand& seed, const VECTOR_XD(DIM)& mean, const MATRIX_XD(DIM)& cov)
          : Normal(seed, boost::normal_distribution<>(0.0, 1.0)), parameters(mean, cov)
      {}
      MultiNormalRandom(const MultiNormalRandom& copy) 
          : Normal(copy.Normal), parameters(copy.parameters)
      {}

      VECTOR_XD(DIM) operator()() {
          VECTOR_XD(DIM) random_number;

          // TODO: check semi-positive definite property for covariances

          Eigen::LLT<MATRIX_XD(DIM)> llt(parameters.covariance);
          MATRIX_XD(DIM) CH = llt.matrixL().transpose();

          for(unsigned i = 0; i < DIM; i++)
              random_number(i) = Normal();

          return (random_number.transpose() * CH).transpose() + parameters.mean;
      }

      MultiNormalRandom operator=(const MultiNormalRandom& mnr) {
          return *this;
      }

  private:
      NormalRandom Normal;
      GaussParam<DIM> parameters;
};


class Random {
  public:
      static UniformRealRandom uniform_real(double min = 0.0, double max = 1.0);

      static NormalRandom gaussian(double mean = 0.0, double variance = 1.0);

      static UniformIntRandom uniform_int(int min = 0, int max = 1);

      template <int DIM>
      static MultiNormalRandom<DIM> multi_gaussian(const VECTOR_XD(DIM)& mean = VECTOR_XD(DIM)::Zero(), const MATRIX_XD(DIM)& cov = MATRIX_XD(DIM)::Identity()) {
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
