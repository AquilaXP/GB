#pragma once

#include <string>
#include <memory>

#include "oml/IModel.h"
#include "oml/Config.h"

namespace oml{

struct OML_API FactoryModel
{
    /// Создать дерево
    static std::unique_ptr<IModel> create( const std::string& name );

    /// Получить имена доступных объектов
    static std::vector< std::string > names();
};

}
