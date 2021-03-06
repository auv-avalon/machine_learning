#ifndef MACHINE_LEARNING__GUASSIAN_PARAMETERS_HPP
#define MACHINE_LEARNING__GUASSIAN_PARAMETERS_HPP

#include <vector>
#include <Eigen/Core>

#define VECTOR_XD(DIM) Eigen::Matrix<double, DIM, 1>
#define MATRIX_XD(DIM) Eigen::Matrix<double, DIM, DIM>

namespace machine_learning {

inline double gaussian1d(double mean, double variance, double x) {
   return exp((-0.5 * (x - mean) * (x - mean)) / variance);
}

template <int DIM>
inline double calc_gaussian(const VECTOR_XD(DIM)& mean, const MATRIX_XD(DIM)& cov, const VECTOR_XD(DIM)& x) {
    double z = (x - mean).transpose() * cov.inverse() * (x - mean);
    return exp(-0.5 * z);
}


template <int DIM>
inline double calc_gaussian_norm(const VECTOR_XD(DIM)& mean, const MATRIX_XD(DIM)& cov, const VECTOR_XD(DIM)& x) {
    return 1.0 / (pow(sqrt(2 * M_PI), mean.rows()) * sqrt(cov.determinant())) 
        * calc_gaussian(mean, cov, x);
}


template <int DIM> 
struct GaussParam {
    VECTOR_XD(DIM) mean;
    MATRIX_XD(DIM) covariance;

    GaussParam(const VECTOR_XD(DIM)& mean = VECTOR_XD(DIM)::Zero(), const MATRIX_XD(DIM)& cov = MATRIX_XD(DIM)::Identity()) 
        : mean(mean), covariance(cov)
    {}

    inline double gaussian(const VECTOR_XD(DIM)& x) {
        return calc_gaussian<DIM>(mean, covariance, x);
    }

    inline double gaussian_norm(const VECTOR_XD(DIM)& x) {
       return calc_gaussian_norm<DIM>(mean, covariance, x);
    }

    inline double mahalanobis(const VECTOR_XD(DIM)& x) {
       return sqrt((x - mean).transpose() * covariance.inverse() * (x - mean));
    }

    inline double euclidean(const VECTOR_XD(DIM)& x) {
       return sqrt((x - mean).transpose() * MATRIX_XD(DIM)::Identity() * (x - mean));
    }
};




template <int DIM>
GaussParam<DIM> calculate_parameters(const std::vector< VECTOR_XD(DIM) >& samples, int cov_scaling = 1) {
    GaussParam<DIM> params(VECTOR_XD(DIM)::Zero(), MATRIX_XD(DIM)::Zero());

    for(unsigned i = 0; i < samples.size(); i++) {
        params.mean += samples[i];
    }

    params.mean /= samples.size();

    for(unsigned i = 0; i < samples.size(); i++) {
        VECTOR_XD(DIM) s = (samples[i] - params.mean);
        params.covariance += s * s.transpose();
    }

    params.covariance /= (samples.size() + cov_scaling);

    return params;
}



}

#endif
