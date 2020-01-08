#pragma once

#include <memory>

#include "oml/ITree.h"
#include "oml/Config.h"

#pragma warning( push )
#pragma warning( disable : 4251 )

namespace oml{

class OML_API RegressionTreeFastMse : public oml::ITree
{
public:
    static std::string get_name();
    RegressionTreeFastMse() = default;
    RegressionTreeFastMse( const RegressionTreeFastMse& ) = delete;
    RegressionTreeFastMse& operator = ( const RegressionTreeFastMse& ) = delete;
    ~RegressionTreeFastMse();

    void init( const nlohmann::json& params ) override;
    void load( const nlohmann::json& tree ) override;
    void uninit() override;
    bool is_init() const override;
    std::string name() const;
    void dump( nlohmann::json& tree ) const override;
    void fit( const data_x_t& X, const data_y_t& y ) override;
    data_y_t predict( const data_x_t& X ) override;
private:
    double predict_1( const std::vector< double >& x );
    void XNoinit() const;

    bool m_is_init = false;

    int m_max_depth = 0;
    int m_min_size = 0;
    int m_averages = 0;

    int m_feature_idx = -1;
    double m_feature_threshold = 0.0;
    double m_value = 0.0;

    std::unique_ptr<RegressionTreeFastMse> left;
    std::unique_ptr<RegressionTreeFastMse> right;
};

}

#pragma warning( pop )
