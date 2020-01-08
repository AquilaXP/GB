#include <iostream>
#include <cmath>

#include "oml/DataSetLoader.h"
#include "oml/GradientBoosting.h"
#include "oml/Util.h"

int main()
{
    oml::DataSetLoader::set_dataset_folder( "../DataSet" );

    auto[ds,t] = oml::DataSetLoader::get_dataset( "digits" );
    
    auto[ds_train, t_train, ds_hold, t_hold] = oml::train_test_split( ds, t, 0.8 );
    
    oml::GradientBoosting tree{ 200, 0.1, 128 };

    tree.fit( ds_train, t_train );
    auto res = tree.predict( ds_hold );
    std::vector< double > r;
    for( auto& rr : res )
        r.emplace_back( std::round( rr ) );

    std::cout << oml::accuracy_score( t_hold, r ) << std::endl;

    return 0;
}
