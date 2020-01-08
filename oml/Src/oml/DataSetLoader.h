#pragma once

#include <filesystem>
#include <string>

#include "oml/Data.h"
#include "oml/Config.h"

namespace oml
{

/*!
    \brief Загрузчик встроенных дата сетов
*/
class OML_API DataSetLoader
{
public:
    /// Установить путь до директории со встроенными дата сетами
    static void set_dataset_folder( const std::filesystem::path& path );

    /// Загрузить доступный дата сет
    static std::pair< data_x_t, data_y_t > get_dataset( const std::string& name );

    /// Получить доступные дата сеты
    static std::vector< std::string > get_names_exists_dataset();
};

}
