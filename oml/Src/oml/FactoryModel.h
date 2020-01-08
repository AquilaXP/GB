#pragma once

#include <string>
#include <memory>

#include "oml/IModel.h"
#include "oml/Config.h"

namespace oml{

struct OML_API FactoryModel
{
    /// ������� ������
    static std::unique_ptr<IModel> create( const std::string& name );

    /// �������� ����� ��������� ��������
    static std::vector< std::string > names();
};

}
