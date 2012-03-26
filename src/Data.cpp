#include "Data.hpp"
#include <stdlib.h>
#include <stdarg.h>
#include <boost/algorithm/string.hpp>
#include <iostream>

namespace machine_learning {

Data::Data(std::istream& inputstream)
{
    inputstream >> *this;
}


Data::Data()
{}

Data::~Data() 
{}

std::ostream& operator<<(std::ostream& out, const Data& data)
{
    out << static_cast<const DataMatrix&>(data);
    return out;
}

std::istream& operator>>(std::istream& in, Data& data) {
    char buffer[1024];
    std::vector<std::string> numbers;
    std::vector<std::string> lines;

    data.resize(1,1);

    while(in.good()) {
        in.getline(buffer, 1024);
        std::string line(buffer);

        boost::trim(line);

        if(line.size() > 0)
            lines.push_back(line);
    }

    boost::split(numbers, lines.front(), boost::is_any_of("\t, "));

    data.resize(lines.size(), numbers.size());

    numbers.clear();

    for(unsigned j = 0; j < lines.size(); j++) {
        boost::split(numbers, lines[j], boost::is_any_of("\t, "));

        for(unsigned i = 0; i < numbers.size(); i++) {
            boost::trim(numbers[i]);
            data(j, i) = atof(numbers[i].c_str());
        }

        numbers.clear();
    }

    return in;
}




std::vector<Data> Data::split(unsigned number) const
{
    std::vector<Data> partitions;

    return partitions;
}

}
