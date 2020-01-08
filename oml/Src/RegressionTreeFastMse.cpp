#include "oml/RegressionTreeFastMse.h"

#include <iterator>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>

#include "oml/Util.h"

namespace oml{

std::string RegressionTreeFastMse::get_name()
{
    return "RegressionTreeFastMse";
}

RegressionTreeFastMse::~RegressionTreeFastMse()
{
    uninit();
}

void RegressionTreeFastMse::init( const nlohmann::json& params )
{
    uninit();
    m_max_depth = 3;
    m_min_size = 3;
    m_averages = 1;

    if( params.contains( "max_depth" ) )
        params.at( "max_depth" ).get_to( m_max_depth );
    if( params.contains( "min_size" ) )
        params.at( "min_size" ).get_to( m_min_size );
    if( params.contains( "averages" ) )
        params.at( "averages" ).get_to( m_averages );
    m_is_init = true;
}

void RegressionTreeFastMse::load( const nlohmann::json& tree )
{
    uninit();

    tree.at( "max_depth" ).get_to( m_max_depth );
    tree.at( "feature_idx" ).get_to( m_feature_idx );
    tree.at( "feature_threshold" ).get_to( m_feature_threshold );
    tree.at( "min_size" ).get_to( m_min_size );
    tree.at( "averages" ).get_to( m_averages );
    tree.at( "value" ).get_to( m_value );
    if( tree.contains( "left" ) )
    {
        left = std::make_unique<RegressionTreeFastMse>();
        left->load( tree.at( "left" ) );
    }
    if( tree.contains( "right" ) )
    {
        right = std::make_unique<RegressionTreeFastMse>();
        right->load( tree.at( "right" ) );
    }
}

void RegressionTreeFastMse::uninit()
{
    m_is_init = false;
    left.reset();
    right.reset();
    m_value = 0.0;
    m_feature_idx = -1;
    m_feature_threshold = 0;
}

bool RegressionTreeFastMse::is_init() const
{
    return m_is_init;
}

std::string RegressionTreeFastMse::name() const
{
    return get_name();
}

void RegressionTreeFastMse::dump( nlohmann::json& tree ) const
{
    if( !is_init() )
        XNoinit();

    tree["max_depth"] = m_max_depth;
    tree["feature_idx"] = m_feature_idx;
    tree["feature_threshold"] = m_feature_threshold;
    tree["min_size"] = m_min_size;
    tree["averages"] = m_averages;
    tree["value"] = m_value;
    if( left )
    {
        nlohmann::json tree_left;
        left->dump( tree_left );
        tree["left"] = std::move( tree_left );
    }
    if( right )
    {
        nlohmann::json tree_right;
        right->dump( tree_right );
        tree["right"] = tree_right;
    }
}

void RegressionTreeFastMse::fit( const data_x_t& X, const data_y_t& y )
{
    if( !is_init() )
        XNoinit();

    // начальное значение - среднее значение y
    m_value = mean( y );
    // начальная ошибка - mse между значением в листе
    double base_error = 0;
    for( auto& v : y )
        base_error += std::pow( v - m_value, 2 );

    double error = base_error;
    bool flag = false;

    // пришли на максимальную глубину
    if( m_max_depth <= 1 )
        return;

    auto dim_shape = X[0].size();

    double left_value = 0.0;
    double right_value = 0.0;

    for( size_t feat = 0; feat < dim_shape; ++feat )
    {
        double prev_error1 = base_error;
        double prev_error2 = 0.0;

        auto idxs = argsort( X, feat );

        // переменные для быстрого преброса суммы
        double mean1 = mean( y );
        double mean2 = 0.0;
        double sm1 = std::accumulate( std::begin( y ), std::end( y ), 0.0 );
        double sm2 = 0.0;

        int N = X.size();
        int N1 = N;
        int N2 = 0;
        int thres = 1;
        while( thres < N - 1 )
        {
            N1 -= 1;
            N2 += 1;

            int idx = idxs[thres];
            double x = X[idx][feat];

            // вычисляем дельты - по ним, в основном, будет делаться переброс
            double delta1 = ( sm1 - y[idx] ) * 1.0 / N1 - mean1;
            double delta2 = ( sm2 + y[idx] ) * 1.0 / N2 - mean2;

            // увеличиваем суммы
            sm1 -= y[idx];
            sm2 += y[idx];

            // пересчитываем ошибки за O( 1 )
            prev_error1 += ( delta1*delta1 ) * N1;
            prev_error1 -= ( y[idx] - mean1 )*( y[idx] - mean1 );
            prev_error1 -= 2 * delta1 * ( sm1 - mean1 * N1 );
            mean1 = sm1 / N1;

            prev_error2 += ( delta2*delta2 ) * N2;
            prev_error2 += ( y[idx] - mean2 )*( y[idx] - mean2 );
            prev_error2 -= 2 * delta2 * ( sm2 - mean2 * N2 );
            mean2 = sm2 / N2;

            // пропускаем близкие друг к другу значения
            if( thres < N - 1 && abs( x - X[idxs[thres + 1]][feat] ) < 1e-5 )
            {
                thres += 1;
                continue;
            }

            if( prev_error1 + prev_error2 < error )
            {
                if( std::min( N1, N2 ) > m_min_size )
                {
                    // переопределяем самый лучший признак и границу по нему
                    m_feature_idx = feat;
                    m_feature_threshold = x;
                    // переопределяем значения в листах
                    left_value = mean1;
                    right_value = mean2;

                    // флаг - значит сделали хороший сплит
                    flag = true;
                    error = prev_error1 + prev_error2;
                }
            }
            thres += 1;
        }
    }

    if( m_feature_idx == -1 )
        return;

    // вызываем потомков дерева
    nlohmann::json param;
    param["max_depth"] = m_max_depth - 1;
    param["min_size"] = m_min_size;
    param["averages"] = m_averages;
    left = std::make_unique<RegressionTreeFastMse>();
    left->init( param );
    left->m_value = left_value;
    right = std::make_unique<RegressionTreeFastMse>();
    right->init( param );
    right->m_value = right_value;

    // новые индексы для обучения потомков
    std::vector< std::vector< double > > leftX;
    std::vector< double > leftY;
    for( size_t i = 0; i < X.size(); ++i )
    {
        auto& v = X[i];
        if( v[m_feature_idx] > m_feature_threshold )
        {
            leftX.push_back( v );
            leftY.push_back( y[i] );
        }
    }
    left->fit( leftX, leftY );

    std::vector< std::vector< double > > rightX;
    std::vector< double > rightY;
    for( size_t i = 0; i < X.size(); ++i )
    {
        auto& v = X[i];
        if( v[m_feature_idx] <= m_feature_threshold )
        {
            rightX.push_back( v );
            rightY.push_back( y[i] );
        }
    }
    right->fit( rightX, rightY );
}

data_y_t RegressionTreeFastMse::predict( const data_x_t& X )
{
    data_y_t y( X.size(), 0.0 );
    for( size_t i = 0; i < X.size(); ++i )
    {
        y[i] = predict_1( X[i] );
    }

    return y;
}

double RegressionTreeFastMse::predict_1( const std::vector< double >& x )
{
    if( m_feature_idx == -1 )
        return m_value;

    if( x[m_feature_idx] > m_feature_threshold )
        return left->predict_1( x );
    else
        return right->predict_1( x );
}

void RegressionTreeFastMse::XNoinit() const
{
    throw oml::Exception( "no init" );
}

}
