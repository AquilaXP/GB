#pragma once

#include <string>

#include "nlohmann/json.hpp"

#include "oml/Data.h"
#include "oml/Config.h"

namespace oml{

    /*
        \brief Базовый класс деревьев решений
    */
    class OML_API ITree
    {
    public:
        ITree() = default;
        ITree( const ITree& ) = delete;
        ITree& operator = ( const ITree& ) = delete;
        virtual ~ITree() = default;

        /// Инициализирован ли
        virtual bool is_init() const = 0;

        /// Инициализация с заданными параметрами
        virtual void init( const nlohmann::json& params ) = 0;

        /// Загрузить обученное дерево
        virtual void load( const nlohmann::json& tree ) = 0;

        /// Сохранить
        virtual void dump( nlohmann::json& tree ) const = 0;

        /// Деинициализирован
        virtual void uninit() = 0;

        /// Обучить
        virtual void fit( const data_x_t& x, const data_y_t& y ) = 0;

        /// Вычислить
        virtual data_y_t predict( const data_x_t& x ) = 0;

        /// Получить имя
        virtual std::string name() const = 0;
    };
}
