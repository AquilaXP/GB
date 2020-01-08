#pragma once

#include <string>
#include <memory>

#include "oml/ITree.h"
#include "oml/Config.h"

namespace oml{

struct OML_API FactoryTree
{
    /// ������� ������
    static std::unique_ptr<ITree> create( const std::string& name );

    /// �������� ����� ��������� ��������
    static std::vector< std::string > names();
};

}
