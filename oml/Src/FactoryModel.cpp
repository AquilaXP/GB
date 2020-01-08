#include "oml/FactoryModel.h"

#include "oml/RegressionTreeFastMse.h"
#include "oml/Exception.h"

std::unique_ptr<oml::IModel> oml::FactoryModel::create( const std::string& name )
{
    std::unique_ptr<oml::IModel> tree;
    if( name == RegressionTreeFastMse::get_name() )
    {
        return std::make_unique<RegressionTreeFastMse>();
    }
    else
    {
        throw oml::Exception( name + " not exists" );
    }

    return tree;
}

std::vector< std::string > oml::FactoryModel::names()
{
    static std::vector< std::string > names = { RegressionTreeFastMse::get_name() };

    return names;
}
