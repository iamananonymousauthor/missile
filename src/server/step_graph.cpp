//
// Created by Anonymous authors on 2023/12/28.
//

#pragma once

#include <vector>
#include <chrono>
#include <future>

#include "../util/common.h"
#include "step_graph.h"
#include "assert.h"

using namespace missilebase;

StepNode::StepNode(ExecutionStepType _type, int _target_idx) :
        prom(),
        future(prom.get_future()) {
    finished.store(false);
    type = _type;
    target_idx = _target_idx;
}

StepNode::StepNode(const ExecutionStepInfo &step_info) :
        prom(),
        future(prom.get_future()) {
    type = step_info.type;
    target_idx = step_info.target_idx;
}

void StepNode::reset_finished() {
    finished.store(false);
    prom = std::promise<bool>();
    future = std::shared_future<bool>(prom.get_future());
}

void StepNode::mark_finished() {
    finished.store(true);
    prom.set_value(true);
}

Status StepNode::wait_for_finish() {
    if(!finished.load(std::memory_order_acquire)) {
        future.get();
    }
    return Status::Succ;
}

bool StepNode::fulfill_prerequisites() {
    for(int i=0; i<prerequisite_node_list.size(); i++) {
        if(!prerequisite_node_list[i]->finished) {
            return false;
        }
    }
    return true;
}

Status StepNode::wait_for_prerequisites() {
    for(int i=0; i<prerequisite_node_list.size(); i++) {
        RETURN_STATUS(prerequisite_node_list[i]->wait_for_finish());
    }
    return Status::Succ;
}

Status StepGraph::build_prerequisites(std::shared_ptr<StepNode> node, ExecutionStepInfo step_info) {
    for(int i=0; i<step_info.prerequisites_list.size(); i++) {
        PrerequisiteStepInfo prerequisite = step_info.prerequisites_list[i];
        if(prerequisite.type == ExecutionStepCreate) {
            if(prerequisite.step_idx >= create_step_node_list.size()) {
                RETURN_STATUS(Status::InvalidArgument);
            }
            node->prerequisite_node_list.push_back(create_step_node_list[prerequisite.step_idx]);
        } else if(prerequisite.type == ExecutionStepSwapIn) {
            if(prerequisite.step_idx >= swap_in_step_node_list.size()) {
                RETURN_STATUS(Status::InvalidArgument);
            }
            node->prerequisite_node_list.push_back(swap_in_step_node_list[prerequisite.step_idx]);
        } else if(prerequisite.type == ExecutionStepSwapOut) {
            if(prerequisite.step_idx >= swap_out_step_node_list.size()) {
                RETURN_STATUS(Status::InvalidArgument);
            }
            node->prerequisite_node_list.push_back(swap_out_step_node_list[prerequisite.step_idx]);
        } else if(prerequisite.type == ExecutionStepLaunch) {
            if(prerequisite.step_idx >= launch_step_node_list.size()) {
                RETURN_STATUS(Status::InvalidArgument);
            }
            node->prerequisite_node_list.push_back(launch_step_node_list[prerequisite.step_idx]);
        } else if(prerequisite.type == ExecutionStepReclaim) {
            if(prerequisite.step_idx >= reclaim_step_node_list.size()) {
                RETURN_STATUS(Status::InvalidArgument);
            }
            node->prerequisite_node_list.push_back(reclaim_step_node_list[prerequisite.step_idx]);
        } else {
            RETURN_STATUS(Status::InvalidArgument);
        }
    }
    return Status::Succ;
}

StepGraph::StepGraph(std::shared_ptr<Model> model, bool is_cold_start) {
    this->model = model;
    ExecutionGraphInfo *graph_info;
    if(is_cold_start) {
        graph_info = &(model->cold_start_execution_graph_info);
    } else {
        graph_info = &(model->warm_start_execution_graph_info);
    }
    assert(create_step_node_list.size() == 0);
    assert(swap_in_step_node_list.size() == 0);
    assert(swap_out_step_node_list.size() == 0);
    assert(launch_step_node_list.size() == 0);
    assert(reclaim_step_node_list.size() == 0);
    for (int step_idx = 0; step_idx < graph_info->create_plan.size(); step_idx++) {
        create_step_node_list.push_back(std::make_shared<StepNode>(graph_info->create_plan[step_idx]));
    }
    for (int step_idx = 0; step_idx < graph_info->swap_in_plan.size(); step_idx++) {
        swap_in_step_node_list.push_back(std::make_shared<StepNode>(graph_info->swap_in_plan[step_idx]));
    }
    for (int step_idx = 0; step_idx < graph_info->swap_out_plan.size(); step_idx++) {
        swap_out_step_node_list.push_back(std::make_shared<StepNode>(graph_info->swap_out_plan[step_idx]));
    }
    for (int step_idx = 0; step_idx < graph_info->launch_plan.size(); step_idx++) {
        launch_step_node_list.push_back(std::make_shared<StepNode>(graph_info->launch_plan[step_idx]));
    }
    for (int step_idx = 0; step_idx < graph_info->reclaim_plan.size(); step_idx++) {
        reclaim_step_node_list.push_back(std::make_shared<StepNode>(graph_info->reclaim_plan[step_idx]));
    }

    // Build prerequisite node pointers
    for (int step_idx = 0; step_idx < graph_info->create_plan.size(); step_idx++) {
        ASSERT_STATUS(this->build_prerequisites(create_step_node_list[step_idx], graph_info->create_plan[step_idx]));
    }
    for (int step_idx = 0; step_idx < graph_info->swap_in_plan.size(); step_idx++) {
        ASSERT_STATUS(this->build_prerequisites(swap_in_step_node_list[step_idx], graph_info->swap_in_plan[step_idx]));
    }
    for (int step_idx = 0; step_idx < graph_info->swap_out_plan.size(); step_idx++) {
        ASSERT_STATUS(this->build_prerequisites(swap_out_step_node_list[step_idx], graph_info->swap_out_plan[step_idx]));
    }
    for (int step_idx = 0; step_idx < graph_info->launch_plan.size(); step_idx++) {
        ASSERT_STATUS(this->build_prerequisites(launch_step_node_list[step_idx], graph_info->launch_plan[step_idx]));
    }
    for (int step_idx = 0; step_idx < graph_info->reclaim_plan.size(); step_idx++) {
        ASSERT_STATUS(this->build_prerequisites(reclaim_step_node_list[step_idx], graph_info->reclaim_plan[step_idx]));
    }
}

void StepGraph::reset() {
    for (int step_idx = 0; step_idx < create_step_node_list.size(); step_idx++) {
        create_step_node_list[step_idx]->reset_finished();
    }
    for (int step_idx = 0; step_idx < swap_in_step_node_list.size(); step_idx++) {
        swap_in_step_node_list[step_idx]->reset_finished();
    }
    for (int step_idx = 0; step_idx < swap_out_step_node_list.size(); step_idx++) {
        swap_out_step_node_list[step_idx]->reset_finished();
    }
    for (int step_idx = 0; step_idx < launch_step_node_list.size(); step_idx++) {
        launch_step_node_list[step_idx]->reset_finished();
    }
    for (int step_idx = 0; step_idx < reclaim_step_node_list.size(); step_idx++) {
        reclaim_step_node_list[step_idx]->reset_finished();
    }
}


bool StepGraph::is_finished(int create_step_offset, 
                            int swap_in_step_offset, 
                            int swap_out_step_offset, 
                            int launch_step_offset, 
                            int reclaim_step_offset) {
    if (create_step_offset < this->create_step_node_list.size()) {
        return false;
    }
    if (swap_in_step_offset < this->swap_in_step_node_list.size()) {
        return false;
    }
    if (swap_out_step_offset < this->swap_out_step_node_list.size()) {
        return false;
    }
    if (launch_step_offset < this->launch_step_node_list.size()) {
        return false;
    }
    if (reclaim_step_offset < this->reclaim_step_node_list.size()) {
        return false;
    }
    return true;
}