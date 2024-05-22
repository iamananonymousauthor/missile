//
// Created by Anonymous authors 2023/12/28.
//

#pragma once

#include <vector>
#include <future>
#include <atomic>

#include "../util/model.h"
#include "../util/common.h"

using namespace missilebase;

class StepNode {
public:
    ExecutionStepType type;
    std::vector<std::shared_ptr<StepNode>> prerequisite_node_list;
    int target_idx;
    std::atomic<bool> finished;

    StepNode(ExecutionStepType _type, int _target_idx);
    StepNode(const ExecutionStepInfo &step_info);

    bool fulfill_prerequisites();
    void reset_finished();
    void mark_finished();

    Status wait_for_finish();
    Status wait_for_prerequisites();

private:
    std::promise<bool> prom;
    std::shared_future<bool> future;
};

class StepGraph {
public:
    StepGraph(std::shared_ptr<Model> model, bool is_cold_start);

    Status build_prerequisites(std::shared_ptr<StepNode> node, ExecutionStepInfo step_info);

    void reset();

    bool is_finished(int create_step_offset, 
                     int swap_in_step_offset, 
                     int swap_out_step_offset, 
                     int launch_step_offset, 
                     int reclaim_step_offset);

    std::vector<std::shared_ptr<StepNode>> create_step_node_list, 
                                           swap_in_step_node_list, 
                                           swap_out_step_node_list, 
                                           launch_step_node_list, 
                                           reclaim_step_node_list;
private:
    std::shared_ptr<Model> model;
};