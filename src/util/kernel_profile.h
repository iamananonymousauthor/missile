#pragma once

namespace missilebase {
namespace utils {
    struct KernelProfile {
        double dram_throughput_perc;
        double compute_sm_perc;
        double duration_ms;
        int max_num_blocks_per_sm;
        double sm_needed;

        KernelProfile() {
            dram_throughput_perc = 0;
            compute_sm_perc = 0;
            duration_ms = 0;
            max_num_blocks_per_sm = 0;
            sm_needed = 0;
        }

        KernelProfile(double _dram_throughput_perc, double _compute_sm_perc, double _duration_ms, int _max_num_blocks_per_sm, double _sm_needed) {
            dram_throughput_perc = _dram_throughput_perc;
            compute_sm_perc = _compute_sm_perc;
            duration_ms = _duration_ms;
            max_num_blocks_per_sm = _max_num_blocks_per_sm;
            sm_needed = _sm_needed;
        }
    };
}
}