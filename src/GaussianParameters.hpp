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

    inline double gaussian(const VECTOR_XD(DIM)& x) {
        return gaussian(mean, covariance, x);
    }

    inline double gaussian_norm(const VECTOR_XD(DIM)& x) {
       return gaussian_norm(mean, covariance, x);
    }
};




template <int DIM>
GaussParam<DIM> calculate_gaussian(const std::vector< VECTOR_XD(DIM) >& samples) {
    GaussParam<DIM> params;

    return params;
}



}

#endif
