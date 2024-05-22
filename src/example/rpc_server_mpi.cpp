#include "server/server.h"

#include <string>
#include <cstdlib>
#include <mpi.h>

std::vector<int> parse_string_to_int_list(std::string str) {
    std::replace(str.begin(), str.end(), ',', ' ');
    std::stringstream iss(str);
    int number;
    std::vector<int> myNumbers;
    std::cout<< "MPS Setting: " << str << " >> [";
    while ( iss >> number ) {
        myNumbers.push_back(number);
        std::cout<<number <<",";
    }
    std::cout << "]" << std::endl;
    return myNumbers;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << std::string(argv[0]) << " scheduler_type cfs_period\n";
        std::cerr << "Example: " << std::string(argv[0]) << " missile 128\n";
        return -1;
    }

    std::string scheduler_type(argv[1]);
    std::vector<int> list_of_mps_percentage = parse_string_to_int_list(std::string(argv[2]));

    MPI_Init(&argc, &argv);
    int mpi_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&mpi_id);
    std::string env_mps_ratio_percentage = std::string("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=")+
                                                       std::to_string(list_of_mps_percentage[mpi_id]);
    if (putenv((char*)env_mps_ratio_percentage.c_str()) != 0) {
        std::cout << "putenv failed" << std::endl;
        exit(1);
    }

    if(scheduler_type.compare("naive") == 0) {
        std::cout << "Scheduler: Naive" << std::endl;
        missilebase::server::MISSILEServer server(missilebase::server::Naive, std::string(DEFAULT_MISSILE_ADDR)+
                                        std::string(":")+std::to_string(DEFAULT_MISSILE_PORT+mpi_id), 128);
        server.run();
        server.wait();
    } else {
        std::cout << "Unsupported scheduler type: " << scheduler_type << std::endl;
    }
    MPI_Finalize();
    return 0;
}