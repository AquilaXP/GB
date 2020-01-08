#pragma once

#include <string>
#include <memory>

#include "oml/ITree.h"
#include "oml/Config.h"

namespace oml{

struct OML_API FactoryTree
{
    /// Создать дерево
    static std::unique_ptr<ITree> create( const std::string& name );

    /// Получить имена доступных объектов
    static std::vector< std::string > names();
};

}
