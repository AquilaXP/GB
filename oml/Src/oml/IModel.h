#pragma once

#include <string>

#include "nlohmann/json.hpp"

#include "oml/Data.h"
#include "oml/Config.h"

namespace oml{

    /*
        \brief ������� ����� �������
    */
    class OML_API IModel
    {
    public:
        IModel() = default;
        IModel( const IModel& ) = delete;
        IModel& operator = ( const IModel& ) = delete;
        virtual ~IModel() = default;

        /// ��������������� ��
        virtual bool is_init() const = 0;

        /// ������������� � ��������� �����������
        virtual void init( const nlohmann::json& params ) = 0;

        /// ��������� ��������� ������
        virtual void load( const nlohmann::json& tree ) = 0;

        /// ���������
        virtual void dump( nlohmann::json& tree ) const = 0;

        /// �����������������
        virtual void uninit() = 0;

        /// �������
        virtual void fit( const data_x_t& x, const data_y_t& y ) = 0;

        /// ���������
        virtual data_y_t predict( const data_x_t& x ) = 0;

        /// �������� ���
        virtual std::string name() const = 0;
    };
}
