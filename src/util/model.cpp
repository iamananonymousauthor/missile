
#include <assert.h>
#include <sstream>
#include "model.h"
#include "common.h"
#include "rapidjson/document.h"

#ifdef __MISSILE_CUDA_GPU__
#include "../cuda_implementation/cuda_impl.h"
#endif

using namespace rapidjson;

void parse_execution_step(const char *execution_step_type, int execution_step_idx, Value &sobj, ExecutionStepInfo &execution_step) {
    assert(sobj.HasMember("target_idx") && sobj["target_idx"].IsInt());
    assert(sobj.HasMember("prerequisites") && sobj["prerequisites"].IsArray());
    execution_step.target_idx = sobj["target_idx"].GetInt();
    for(auto &preinfo : sobj["prerequisites"].GetArray()) {
        ASSERT_MSG(preinfo.HasMember("type") && preinfo["type"].IsString(), "at " << execution_step_type << " -> " << execution_step_idx);
        ASSERT_MSG(preinfo.HasMember("step_idx") && preinfo["step_idx"].IsInt(), "at " << execution_step_type << " -> " << execution_step_idx);
        std::string prerequisite_type = preinfo["type"].GetString();
        PrerequisiteStepInfo pre_step_info;
        if(prerequisite_type.compare("create") == 0) {
            pre_step_info.type = ExecutionStepCreate;
        } else if(prerequisite_type.compare("swap_in") == 0) {
            pre_step_info.type = ExecutionStepSwapIn;
        } else if(prerequisite_type.compare("swap_out") == 0) {
            pre_step_info.type = ExecutionStepSwapOut;
        } else if(prerequisite_type.compare("launch") == 0) {
            pre_step_info.type = ExecutionStepLaunch;
        } else if(prerequisite_type.compare("reclaim") == 0) {
            pre_step_info.type = ExecutionStepReclaim;
        } else {
            pre_step_info.type = ExecutionStepInvalid;
        }
        pre_step_info.step_idx = preinfo["step_idx"].GetInt();
        execution_step.prerequisites_list.push_back(pre_step_info);
    }
}

void parse_execution_graph(Value &eobj, ExecutionGraphInfo &graph_info) {
    assert(eobj.HasMember("create"));
    assert(eobj["create"].IsArray());
    for (auto &sobj : eobj["create"].GetArray()) {
        ExecutionStepInfo execution_step_info;
        parse_execution_step("create", graph_info.create_plan.size(), sobj, execution_step_info);
        graph_info.create_plan.push_back(execution_step_info);
    }

    assert(eobj.HasMember("swap_in"));
    assert(eobj["swap_in"].IsArray());
    for (auto &sobj : eobj["swap_in"].GetArray()) {
        ExecutionStepInfo execution_step_info;
        parse_execution_step("swap_in", graph_info.swap_in_plan.size(), sobj, execution_step_info);
        graph_info.swap_in_plan.push_back(execution_step_info);
    }

    assert(eobj.HasMember("swap_out"));
    assert(eobj["swap_out"].IsArray());
    for (auto &sobj : eobj["swap_out"].GetArray()) {
        ExecutionStepInfo execution_step_info;
        parse_execution_step("swap_out", graph_info.swap_out_plan.size(), sobj, execution_step_info);
        graph_info.swap_out_plan.push_back(execution_step_info);
    }

    assert(eobj.HasMember("launch"));
    assert(eobj["launch"].IsArray());
    for (auto &sobj : eobj["launch"].GetArray()) {
        ExecutionStepInfo execution_step_info;
        parse_execution_step("launch", graph_info.launch_plan.size(), sobj, execution_step_info);
        graph_info.launch_plan.push_back(execution_step_info);
    }

    assert(eobj.HasMember("reclaim"));
    assert(eobj["reclaim"].IsArray());
    for (auto &sobj : eobj["reclaim"].GetArray()) {
        ExecutionStepInfo execution_step_info;
        parse_execution_step("reclaim", graph_info.reclaim_plan.size(), sobj, execution_step_info);
        graph_info.reclaim_plan.push_back(execution_step_info);
    }
}

Model* Model::from_json(const char* json_file, bool load_execution_plan) {
    std::ifstream fs(json_file);
    std::string tmp, str = "";

    while (getline(fs, tmp)) str += tmp;
    fs.close();

    Document jobj;
    jobj.Parse(str.c_str());

    Model* m = new Model;
    assert(jobj.HasMember("storage"));
    const Value& storage_list = jobj["storage"];
    assert(storage_list.IsArray());
    for (auto &storage_entry : storage_list.GetArray()) {
        assert(storage_entry.HasMember("name") && storage_entry["name"].IsString());
        assert(storage_entry.HasMember("size") && storage_entry["size"].IsInt64());
        assert(storage_entry.HasMember("stype") && storage_entry["stype"].IsString());
        assert(storage_entry.HasMember("need_spt") && storage_entry["need_spt"].IsBool());
        m->storage.push_back(StorageInfo{
            storage_entry["name"].GetString(),
            storage_entry["size"].GetInt64(),
            storage_entry["stype"].GetString(),
            storage_entry["need_spt"].GetBool()
        });
    }

    assert(jobj.HasMember("ops"));
    const Value& op_list = jobj["ops"];
    assert(op_list.IsArray());
    for (auto &opinfo : op_list.GetArray()) {
        assert(opinfo.HasMember("name") && opinfo["name"].IsString());
        assert(opinfo.HasMember("input_storage_idx") && opinfo["input_storage_idx"].IsArray());
        assert(opinfo.HasMember("output_storage_idx") && opinfo["output_storage_idx"].IsArray());
        assert(opinfo.HasMember("kernels") && opinfo["kernels"].IsArray());
        OpInfo op;
        op.name = opinfo["name"].GetString();
        for(auto &input_storage_idx : opinfo["input_storage_idx"].GetArray()) {
            assert(input_storage_idx.IsInt());
            op.input_storage_idx.push_back(input_storage_idx.GetInt());
        }
        for(auto &output_storage_idx : opinfo["output_storage_idx"].GetArray()) {
            op.output_storage_idx.push_back(output_storage_idx.GetInt());
        }
        for(auto &kinfo : opinfo["kernels"].GetArray()) {
            KernelInfo k;

            assert(kinfo.HasMember("name") && kinfo["name"].IsString());
            assert(kinfo.HasMember("args") && kinfo["args"].IsArray());
            assert(kinfo.HasMember("launch_params") && kinfo["launch_params"].IsArray());

            k.name = kinfo["name"].GetString();
            for (auto &arg : kinfo["args"].GetArray()) {
                assert(arg.IsInt());
                k.args.push_back(arg.GetInt());
            }

            auto launch_params_array = kinfo["launch_params"].GetArray();
            assert(launch_params_array.Size() == 6);
            for (int i = 0; i < 6; i++) {
                assert(launch_params_array[i].IsInt());
                k.launch_params[i] = launch_params_array[i].GetInt();
            }
            op.kernels.push_back(k);
        }
        m->ops.push_back(op);
    }

    assert(jobj.HasMember("args") && jobj["args"].IsArray());
    for (auto &arg : jobj["args"].GetArray()) {
        m->args.push_back(arg.GetInt());
    }

    if(load_execution_plan) {
        assert(jobj.HasMember("cold_start_execution_plan") && jobj["cold_start_execution_plan"].IsObject());
        parse_execution_graph(jobj["cold_start_execution_plan"], m->cold_start_execution_graph_info);

        assert(jobj.HasMember("warm_start_execution_plan") && jobj["warm_start_execution_plan"].IsObject());
        parse_execution_graph(jobj["warm_start_execution_plan"], m->warm_start_execution_graph_info);
    }
    return m;
}

size_t Model::get_stype_size(std::string &stype) {
    if (stype == "float32") return 4;
    if (stype == "int64") return 8;
    if (stype == "byte") return 1;
    if (stype == "uint1") return 1;
    if (stype == "int32") return 4;
    std::cout << stype << " is undefined" << std::endl;
    assert(false);
    return 0;
}