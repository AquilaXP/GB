#pragma once

#include <memory>
#include <vector>

#include "nlohmann/json.hpp"

#include "oml/IModel.h"
#include "oml/Config.h"

namespace oml{

class OML_API GradientBoosting
{
    size_t m_n_estimators = 0;
    int m_max_depth = 0;
    double m_learning_rate = 0;
    int m_min_size = 0;
    std::vector< double > m_loss_by_iter;
    int m_n_samples = 0;

    std::vector< std::unique_ptr<oml::IModel> > trees;
    std::vector< std::vector< double > > m_X;
    std::vector< double > m_y;
public:
    GradientBoosting(
        int n_estimators = 100, double learning_rate = 0.1, int max_depth = 3,
        int random_state = 17, int n_samples = 15, int min_size = 5 );
    GradientBoosting( const GradientBoosting& other ) = delete;
    GradientBoosting& operator = ( const GradientBoosting& ) = delete;

    void init( int n_estimators = 100, double learning_rate = 0.1, int max_depth = 3,
        int random_state = 17, int n_samples = 15, int min_size = 5 );
    void uninit();
    void init_ready( const nlohmann::json& json );
    void dump( nlohmann::json& json );
    void fit( const data_x_t& X, const data_y_t& y );

    std::vector< double > predict( const std::vector< std::vector< double > >& X );

    std::vector< double > get_loss() const;

private:
    std::vector< double > initialization( const std::vector< double >& y );
};

}
