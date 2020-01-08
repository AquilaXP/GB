#pragma once

#include <stdexcept>

#include "oml/Exception.h"

namespace oml{

class Exception : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

}
