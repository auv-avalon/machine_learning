#ifndef _MACHINE_LEARNING_DATA_HPP_
#define _MACHINE_LEARNING_DATA_HPP_

#include <Eigen/Core>
#include <vector>
#include <iostream>

namespace machine_learning {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> DataMatrix;

class Data : public DataMatrix {
 public:
    friend std::ostream& operator<<(std::ostream& out, const Data& data);
    friend std::istream& operator>>(std::istream& in, Data& data);

    Data();
    Data(const DataMatrix& matrix) : DataMatrix(matrix) {}
    ~Data();

    std::vector<Data> split(unsigned partitions) const;
};

} // namespace machine_learning

#endif
