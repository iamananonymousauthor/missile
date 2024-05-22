#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <unordered_map>

class StorageInfo {
public:
    std::string name;
    uint64_t size; // Note: in terms of number of elements
    std::string stype;
    bool need_spt; // Only memory-bound tensors need to be applied with Shadow Page Table (SPT)
    uint64_t alloced_size_bytes;
};

class KernelInfo {
public:
    std::string name;
    uint32_t launch_params[6];
    std::vector<size_t> args;
};

class OpInfo {
public:
    std::string name;
    std::vector<uint32_t> input_storage_idx, output_storage_idx;
    std::vector<KernelInfo> kernels;
};

enum ExecutionStepType {
    ExecutionStepCreate,
    ExecutionStepSwapIn,
    ExecutionStepSwapOut,
    ExecutionStepLaunch,
    ExecutionStepReclaim,
    ExecutionStepInvalid
};

class PrerequisiteStepInfo {
public:
    ExecutionStepType type;
    int step_idx;
};

class ExecutionStepInfo {
public:
    ExecutionStepType type;
    // If this is a launch step, target_idx = opeartor idx
    // Otherwise, target_idx = storage idx
    int target_idx;
    std::vector<PrerequisiteStepInfo> prerequisites_list;
};

class ExecutionGraphInfo {
public:
    std::vector<ExecutionStepInfo> create_plan, swap_in_plan, swap_out_plan, launch_plan, reclaim_plan;
};

class Model {
public:
    std::vector<StorageInfo> storage;
    std::vector<OpInfo> ops;
    std::vector<uint32_t> args;
    //std::unordered_map<std::string, size_t> shared_memory;
    ExecutionGraphInfo cold_start_execution_graph_info, warm_start_execution_graph_info;

public:
    static Model* from_json(const char* json_file, bool load_execution_plan);
    static size_t get_stype_size(std::string &stype);
};

struct ParamData {
    void* h_ptr;
    uint64_t size;
};

typedef std::unordered_map<std::string, ParamData> ModelParam;

class ModelParamParser {
public:
    static ModelParam* parse_from_file(const char* param_file);
};