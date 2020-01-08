#pragma once

#include <numeric>
#include <tuple>
#include <random>
#include <limits>

#include "oml/Data.h"
#include "oml/Exception.h"

namespace oml{

/// Расчите среднего
inline double mean( const data_y_t& arr )
{
    auto res = std::accumulate( arr.begin(), arr.end(), 0.0 );
    return res / arr.size();
}

/// Выдает массив индексов, отсортированных по возрастанию
/*!
    \brief  Выдает массив индексов отсортированных по возрастанию
    \param  data     данные
    \param  column   номер столбца, по которой выполняется сортировка
    \return индексы по возрастанию
*/
inline std::vector<size_t> argsort( const data_x_t& data, size_t column )
{
    if( data.empty() )
        return std::vector<size_t>();

    if( data[0].size() <= column )
        throw Exception( "incorect param column" );

    std::vector < std::pair< double, size_t > > d( data.size() );
    for( size_t i = 0; i < d.size(); ++i )
        d[i] = std::make_pair( data[i][column], i );

    std::sort( d.begin(), d.end(),
        []( const auto& a, const auto& b ) -> bool
    {
        return a.first < b.first;
    } );

    std::vector< size_t > indexs( d.size() );
    for( size_t i = 0; i < indexs.size(); ++i )
        indexs[i] = d[i].second;

    return indexs;
}

/// Рассчитывает средне квардатичную ошибку
inline double mse( const data_y_t& y1, const data_y_t& y2 )
{
    if( y1.size() != y2.size() )
        throw Exception( "not equals size, y1.size() != y2.size()" );

    double res = 0.0;
    for( size_t i = 0; i < y1.size(); ++i )
    {
        double d = y1[i] - y2[i];
        res += d * d;
    }
    return res;
}

/*!
    \brief Разбивает дата сет на обучаующую и проверочную выборку
    \param  x           датасет
    \param  y           target
    \param  test_size   размер обучающей выборки (0; 1)
    \param  begin_state начальное состояние рандоматора
    \return возращает <train_data_set,train_target, hold_data_set, hold_target>
*/
inline std::tuple< data_x_t, data_y_t, data_x_t, data_y_t > train_test_split(  const data_x_t& x, const data_y_t& y, double train_size, size_t begin_state = 0 )
{
    if( x.size() != y.size() )
        throw oml::Exception( "not equales size, x.size() != y.size()" );

    if( x.empty() )
        throw oml::Exception( "empty data" );

    if( 0.0 >= train_size && train_size >= 1.0 )
        throw oml::Exception( "incorect precent test data" );

    std::mt19937_64 mt64( begin_state );
    std::vector< size_t > indexs( x.size(), 0 );
    std::iota( std::begin( indexs ), std::end( indexs ), 0 );
    std::shuffle( std::begin( indexs ), std::end( indexs ), mt64 );
    size_t size = static_cast<size_t>( train_size * x.size() );
    data_x_t trainX( size );
    data_y_t trainY( size );

    for( size_t i = 0; i < size; ++i )
    {
        trainX[i] = x[indexs[i]];
        trainY[i] = y[indexs[i]];
    }

    data_x_t holdoutX;
    data_y_t holdoutY;
    for( size_t i = size; i < x.size(); ++i )
    {
        holdoutX.emplace_back( x[indexs[i]] );
        holdoutY.emplace_back( y[indexs[i]] );
    }

    return std::make_tuple( trainX, trainY, holdoutX, holdoutY );
}

/// Расчитывает схожесть результатов
inline double accuracy_score( const data_y_t& etalon, const data_y_t& res )
{
    if( etalon.size() != res.size() )
        throw Exception( "not equales size" );

    double sumOk = 0;
    for( size_t i = 0; i < res.size(); ++i )
        if( etalon[i] - res[i] < std::numeric_limits<double>::epsilon() )
            sumOk += 1;

    return sumOk / etalon.size();
}

}
