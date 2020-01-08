#include "oml/GradientBoosting.h"

#include "oml/FactoryTree.h"
#include "oml/Util.h"

template< class Ty, class Alloc >
void full_clear( std::vector< Ty, Alloc >& vec )
{
    std::vector< Ty, Alloc >().swap( vec );
}

namespace oml{

GradientBoosting::GradientBoosting( int n_estimators /*= 100*/, double learning_rate /*= 0.1*/, int max_depth /*= 3*/, int random_state /*= 17*/, int n_samples /*= 15*/, int min_size /*= 5 */ )
{
    init( n_estimators, learning_rate, max_depth, random_state, n_samples, min_size );
}

void GradientBoosting::init( int n_estimators /*= 100*/, double learning_rate /*= 0.1*/, int max_depth /*= 3*/, int random_state /*= 17*/, int n_samples /*= 15*/, int min_size /*= 5 */ )
{
    m_n_estimators = n_estimators;
    m_max_depth = max_depth;
    m_learning_rate = learning_rate;
    m_min_size = min_size;
    m_n_samples = n_samples;
}

void GradientBoosting::uninit()
{
    full_clear( trees );
    full_clear( m_X );
    full_clear( m_y );
    full_clear( m_loss_by_iter );
}

void GradientBoosting::init_ready( const nlohmann::json& json )
{
    uninit();

    json.at( "n_estimators" ).get_to( m_n_estimators );
    json.at( "max_depth" ).get_to( m_max_depth );
    json.at( "min_size" ).get_to( m_min_size );
    json.at( "n_samples" ).get_to( m_n_samples );
    json.at( "learning_rate" ).get_to( m_learning_rate );
    json.at( "loss_by_iter" ).get_to( m_loss_by_iter );
    json.at( "y" ).get_to( m_y );
    std::string type_tree;
    json.at( "type_tree" ).get_to( type_tree );
    std::vector< nlohmann::json > tree_json;
    json.at( "trees" ).get_to( tree_json );
    for( const auto& json : tree_json )
    {
        auto tree = FactoryTree::create( type_tree );
        tree->load( json );
        trees.emplace_back( std::move( tree ) );
    }
}

void GradientBoosting::dump( nlohmann::json& json )
{
    json["n_estimators"] = m_n_estimators;
    json["max_depth"] = m_max_depth;
    json["min_size"] = m_min_size;
    json["n_samples"] = m_n_samples;
    json["learning_rate"] = m_learning_rate;
    json["loss_by_iter"] = m_loss_by_iter;
    json["type_tree"] = trees[0]->name();
    std::vector< nlohmann::json > tree_json;
    for( auto& tree : trees )
    {
        nlohmann::json json;
        tree->dump( json );
        tree_json.emplace_back( std::move( json ) );
    }
    json["trees"] = tree_json;
    json["y"] = m_y;
}

void GradientBoosting::fit( const oml::data_x_t& X, const data_y_t& y )
{
    m_X = X;
    m_y = y;
    full_clear( trees );
    auto b = initialization( y );

    double  mean_y = mean( y );

    auto prediction = b;
    for( size_t t = 0; t < m_n_estimators; ++t )
    {
        std::vector< double > resid = y;
        if( t != 0 )
        {
            // антиградиент
            for( size_t i = 0; i < y.size(); ++i )
            {
                resid[i] = y[i] - prediction[i];
            }
        }

        auto tree = FactoryTree::create( "RegressionTreeFastMse" );
        nlohmann::json param_tree;
        param_tree["max_depth"] = m_max_depth;
        param_tree["min_size"] = m_min_size;
        tree->init( param_tree );

        // обучаемся на векторе антиградиента
        tree->fit( X, resid );
        b = tree->predict( X );
        trees.emplace_back( std::move( tree ) );
        for( size_t i = 0; i < prediction.size(); ++i )
        {
            prediction[i] += m_learning_rate * b[i];
        }
        if( t > 0 )
        {
            m_loss_by_iter.emplace_back( mse( y, prediction ) );
        }
    }
}

std::vector< double > GradientBoosting::predict( const std::vector< std::vector< double > >& X )
{
    std::vector< double > pred( X.size(), mean( m_y ) );
    for( size_t t = 0; t < m_n_estimators; ++t )
    {
        auto b = trees[t]->predict( X );
        std::transform( pred.begin(), pred.end(), b.begin(), pred.begin(),
            [&]( double pred_v, double b_v ){ return pred_v + m_learning_rate * b_v; } );
    }

    return pred;
}

std::vector< double > GradientBoosting::initialization( const std::vector< double >& y )
{
    return std::vector< double >( y.size(), mean( y ) );
}

}
