#include "oml/DataSetLoader.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <iterator>
#include <vector>
#include <utility>

#include "oml/Exception.h"

namespace fs = std::filesystem;

namespace oml{

fs::path folder_dataset;

auto read_data_set( const std::string& path )
{
    data_x_t data_set;
    std::ifstream istr( path );
    std::string line;
    while( std::getline( istr, line ) )
    {
        if( line.empty() )
            continue;

        std::string skip;
        std::istringstream ss( line );
        std::vector< double > data_line;
        std::copy( std::istream_iterator<double>( ss ), std::istream_iterator<double>(), std::back_insert_iterator( data_line ) );
        data_set.emplace_back( std::move( data_line ) );
    }

    return data_set;
}


auto read_target( const std::string& path )
{
    std::ifstream istr( path );
    std::vector< double > target_data;

    std::copy( std::istream_iterator<double>( istr ), std::istream_iterator<double>(), std::back_insert_iterator( target_data ) );

    return target_data;
}

void DataSetLoader::set_dataset_folder( const std::filesystem::path& path )
{
    folder_dataset = path;
}

std::pair< data_x_t, data_y_t > DataSetLoader::get_dataset( const std::string& name )
{
    fs::path dataset_path = folder_dataset / ( name + ".dataset" );
    fs::path target_path = folder_dataset / ( name + ".target" );

    if( !fs::exists( dataset_path ) )
        throw Exception( "not exists dataset with name:" + name );
    if( !fs::exists( target_path ) )
        throw Exception( "not exists target with name:" + name );

    data_x_t ds = read_data_set( dataset_path.string() );
    data_y_t t = read_target( target_path.string() );

    return std::make_pair( std::move( ds ), std::move( t ) );
}

std::vector< std::string > DataSetLoader::get_names_exists_dataset()
{
    fs::directory_iterator curr( folder_dataset );
    fs::directory_iterator end;
    std::vector< std::string > names;

    for( ; curr != end; ++curr )
    {
        if( fs::is_regular_file( *curr ) && curr->path().extension() == ".dataset" )
        {
            names.emplace_back( curr->path().stem().string() );
        }
    }

    return names;
}

}
