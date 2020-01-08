#include "oml/FactoryTree.h"

#include "oml/RegressionTreeFastMse.h"
#include "oml/Exception.h"

std::unique_ptr<oml::ITree> oml::FactoryTree::create( const std::string& name )
{
    std::unique_ptr<oml::ITree> tree;
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

std::vector< std::string > oml::FactoryTree::names()
{
    static std::vector< std::string > names = { RegressionTreeFastMse::get_name() };

    return names;
}
