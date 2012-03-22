fndef MACHINE_LEARNING__GUASSIAN_PARAMETERS_HPP
#define MACHINE_LEARNING__GUASSIAN_PARAMETERS_HPP

#define VECTOR_XD(DIM) Eigen::Matrix<double, DIM, 1, Eigen::DontAlign>
#define MATRIX_XD(DIM) Eigen::Matrix<double, DIM, DIM, Eigen::DontAlign>

namespace machine_learning {

template <int DIM> 
struct GaussParam {
    VECTOR_XD(DIM) mean;
    MATRIX_XD(DIM) covariance;
};



template <int DIM>
struct EigenValues {
    VECTOR_XD(DIM) x;
    double alpha;
};



template <int DIM>
double gaussian(const GaussParam<DIM>& param, const VECTOR_XD(DIM)& x) {
    double y = (x - param.mean);
    return exp(-0.5 * y.transpose() * param.covariance.inverse() * y);
}


template <int DIM>
inline double gaussian_norm(const GaussParam<DIM>& param, const VECTOR_XD(DIM)& x) {
    return 1.0 / (pow(sqrt(2 * M_PI), param.mean.rows()) * sqrt(param.covariance.determinat())) 
        * gaussian(param, x);
}



template<int DIM>
std::vector<EigenValues<DIM> > computeEigen(const GaussParam<DIM>& param) {
    std::vector<EigenValues<DIM> > values;

    return values;
};



}

#endif
