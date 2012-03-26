#ifndef MACHINE_LEARNING__GUASSIAN_PARAMETERS_HPP
#define MACHINE_LEARNING__GUASSIAN_PARAMETERS_HPP

#include <vector>

#define VECTOR_XD(DIM) Eigen::Matrix<double, DIM, 1, Eigen::DontAlign>
#define MATRIX_XD(DIM) Eigen::Matrix<double, DIM, DIM, Eigen::DontAlign>

namespace machine_learning {


template <int DIM>
inline double gaussian(const VECTOR_XD(DIM)& mean, const MATRIX_XD(DIM)& cov, const VECTOR_XD(DIM)& x) {
    return exp(-0.5 * (x - mean).transpose() * cov.inverse() * (x - mean));
}


template <int DIM>
inline double gaussian_norm(const VECTOR_XD(DIM)& mean, const MATRIX_XD(DIM)& cov, const VECTOR_XD(DIM)& x) {
    return 1.0 / (pow(sqrt(2 * M_PI), mean.rows()) * sqrt(cov.determinat())) 
        * gaussian(mean, cov, x);
}


template <int DIM> 
struct GaussParam {
    VECTOR_XD(DIM) mean;
    MATRIX_XD(DIM) covariance;

    GaussParam(const VECTOR_XD(DIM)& mean, const MATRIX_XD(DIM)& cov) 
        : mean(mean), covariance(cov)
    {}

    inline double gaussian(const VECTOR_XD(DIM)& x) {
        return gaussian(mean, covariance, x);
    }

    inline double gaussian_norm(const VECTOR_XD(DIM)& x) {
       return gaussian_norm(mean, covariance, x);
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
