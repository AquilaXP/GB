#pragma once

#include <filesystem>
#include <string>

#include "oml/Data.h"
#include "oml/Config.h"

namespace oml
{

/*!
    \brief ��������� ���������� ���� �����
*/
class OML_API DataSetLoader
{
public:
    /// ���������� ���� �� ���������� �� ����������� ���� ������
    static void set_dataset_folder( const std::filesystem::path& path );

    /// ��������� ��������� ���� ���
    static std::pair< data_x_t, data_y_t > get_dataset( const std::string& name );

    /// �������� ��������� ���� ����
    static std::vector< std::string > get_names_exists_dataset();
};

}
