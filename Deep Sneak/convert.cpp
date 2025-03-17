

#include "stdafx.h"
#include "pch.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#ifdef ARDUINO_ETC
#include "filesystem.h"
#else
#include <filesystem>
#endif
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include "Windows.h"
#define NOMINMAX
#if 1
#include "torch/torch.h"
#include <torch/script.h>
#include <torch/serialize.h>
#endif

//#include "compatibility.h"
#include "convert.h"

// NOTE: For full functionality, a complete safetensors library implementation is required.
// Below is a minimal dummy implementation to mimic safe_open and save_file functionalities.
// In a production setting, link against an actual safetensors C++ library.

namespace fs = std::filesystem;

#ifndef PROPELLER
std::unordered_map<std::string, std::pair<std::string, int>> mapping = {
	{"embed_tokens",{"embed",0}},
	{"input_layernorm",{"attn_norm", -1}},
	{"post_attention_layernorm", {"ffn_norm", -1}},
    {"q_proj", {"wq", 0}},
    {"q_a_proj", {"wq_a", -1}},
    {"q_a_layernorm", {"q_norm", -1}},
    {"q_b_proj", {"wq_b", 0}},
    {"kv_a_proj_with_mqa", {"wkv_a", -1}},
    {"kv_a_layernorm", {"kv_norm", -1}},
    {"kv_b_proj", {"wkv_b", 0}},
    {"o_proj", {"wo", 1}},
    {"gate", {"gate", -1}},
    {"gate_proj", {"w1", 0}},
    {"down_proj", {"w2", 1}},
    {"up_proj", {"w3", 0}},
    {"norm", {"norm", -1}},
    {"lm_head", {"head", 0}},
    {"scale", {"scale", -1}},
};
#else

// Mapping dictionary: mapping keys to corresponding new key and dimension. 
// For None in Python, we use -1 in C++.

std::unordered_map<std::string, std::pair<std::string, int>> mapping [] = {
	&map_item("embed_tokens","embed",0),
    &map_item("input_layernorm","attn_norm", -1),
    &map_item("post_attention_layernorm","ffn_norm", -1),
    &map_item("q_proj", "wq", 0),
    &map_item("q_a_proj", "wq_a", -1),
    &map_item("q_a_layernorm","q_norm", -1),
    &map_item("q_b_proj","wq_b", 0),
    &map_item("kv_a_proj_with_mqa","wkv_a", -1),
    &map_item("kv_a_layernorm", "kv_norm", -1),
    &map_item("kv_b_proj","wkv_b", 0),
    &map_item("o_proj", "wo", 1),
    &map_item("gate", "gate", -1),
    &map_item("gate_proj", "w1", 0),
    &map_item("down_proj", "w2", 1),
    &map_item("up_proj", "w3", 0),
    &map_item("norm", "norm", -1),
    &map_item("lm_head", "head", 0),
    &map_item("scale","scale", -1),
};
#endif

// Minimal command-line argument parser.
struct Arguments {
    std::string hf_ckpt_path;
    std::string save_path;
    int n_experts;
    int model_parallel;
};

Arguments parse_args(int argc, char* argv[]);
void main_conversion(const std::string& hf_ckpt_path, const std::string& save_path, int n_experts, int mp);

int convert::main(int argc, char* argv[])
{
    // Parse command-line arguments.
    Arguments args = parse_args(argc, argv);
    // Assert that n_experts is divisible by model_parallel.
    assert(args.n_experts % args.model_parallel == 0);
    // Call the main conversion function.
    main_conversion(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel);
    return 0;
}

void fnDeepSeekConvert1()
{

}

// Helper function: Replaces all occurrences of 'from' with 'to' in 'str'
std::string replace_all(const std::string &str, const std::string &from, const std::string &to) {
    if(from.empty())
        return str;
    std::string result = str;
    size_t start_pos = 0;
    while((start_pos = result.find(from, start_pos)) != std::string::npos) {
        result.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return result;
}

// Helper function: Splits string by delimiter.
std::vector<std::string> split_string(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

// Class to mimic safetensors.torch safe_open functionality.
class SafeTensorFile {
public:
    // Constructor that "opens" the file given the filename.
    SafeTensorFile(const std::string &filename) {
        // Dummy implementation: in a real scenario, load the safetensors file and parse its contents.
        // Here we simulate with torch::load if needed, but we leave the internal state empty.
        file_name = filename;
        // For demonstration, we load a dummy state dict if file exists.
        // This is only to allow the code to run without actual safetensors implementation.
        // In practice, replace this with actual safetensors file parsing.
        // We use torch::serialize::InputArchive to mimic key extraction.
        try {
            torch::serialize::InputArchive archive;
            archive.load_from(filename);
            // The keys are not directly extractable, so we leave keys empty.
            // In an actual implementation, keys and tensors would be read.
        } catch (...) {
            // If loading fails, we simply leave the keys empty.
        }
    }

    // Returns all tensor keys in the file.
    std::vector<std::string> keys() 
	{
        // Dummy implementation: should return the actual keys from the safetensors file.
        // For demonstration, we return an empty vector.
		std::vector<std::string> result;
        return result;
    }

    // Retrieves the tensor corresponding to the given key.
    torch::Tensor get_tensor(const std::string &name) {
        // Dummy implementation: in a real implementation, retrieve the tensor from the file.
        // Here we simulate by returning an empty tensor.
//        return torch::empty({0});
		return torch::empty(0);
    }
private:
    std::string file_name;
};

// Mimics the safe_open context manager from safetensors.torch.
// The caller is responsible for deleting the returned pointer.
SafeTensorFile* safe_open(const std::string &filename, const std::string &framework, const std::string &device) {
    // framework and device parameters are ignored in this dummy implementation.
    return new SafeTensorFile(filename);
}

// Mimics the save_file function from safetensors.torch.
// Saves the given state dictionary to the specified filename.
void save_file(const std::unordered_map<std::string, torch::Tensor>& state, const std::string &filename) {
    // We use torch::serialize::OutputArchive to mimic saving.
    torch::serialize::OutputArchive archive;
    for (const auto &item : state) {
        archive.write(item.first, item.second);
    }
    archive.save_to(filename);
}

/*
Converts and saves model checkpoint files into a specified format.

Args:
    hf_ckpt_path (std::string): Path to the directory containing the input checkpoint files.
    save_path (std::string): Path to the directory where the converted checkpoint files will be saved.
    n_experts (int): Total number of experts in the model.
    mp (int): Model parallelism factor.
    
Returns:
    void
*/
void main_conversion(const std::string &hf_ckpt_path, const std::string &save_path, int n_experts, int mp) {
    // Set the number of threads for torch.
    torch::set_num_threads(8);
    int n_local_experts = n_experts / mp;
    // Create a vector of state dictionaries (each is an unordered_map<string, torch::Tensor>) for each model parallel shard.
    std::vector<std::unordered_map<std::string, torch::Tensor>> state_dicts(mp);
    
    // Iterate over all .safetensors files in hf_ckpt_path.
    for (const auto &entry : fs::directory_iterator(hf_ckpt_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            std::string file_path = entry.path().string();
            // Progress display similar to tqdm.
            std::cout << "Processing file: " << file_path << std::endl;

            SafeTensorFile* f = safe_open(file_path, "pt", "cpu");
            // Get tensor keys from the file.
            std::vector<std::string> keys = f->keys();
            for (std::string name : keys) {
                // If "model.layers.61" is in the name, skip this tensor.
                if (name.find("model.layers.61") != std::string::npos) {
                    continue;
                }
                // Get the tensor associated with the key.
                torch::Tensor param = f->get_tensor(name);
                // If the name starts with "model.", remove that prefix.
                if (name.rfind("model.", 0) == 0) {
                    name = name.substr(6);
                }
                // Replace substrings in the name.
                name = replace_all(name, "self_attn", "attn");
                name = replace_all(name, "mlp", "ffn");
                name = replace_all(name, "weight_scale_inv", "scale");
                name = replace_all(name, "e_score_correction_bias", "bias");
                // Split the name by '.' to extract parts.
                std::vector<std::string> parts = split_string(name, '.');
                // Get the key which is the second last element.
                if (parts.size() < 2) {
                    throw std::runtime_error("Invalid tensor name format: " + name);
                }
                std::string key = parts[parts.size() - 2];
                // Ensure the key exists in the mapping.
                if (mapping.find(key) == mapping.end()) {
                    throw std::runtime_error("Key " + key + " not found in mapping.");
                }
                // Retrieve new key and dimension.
                std::string new_key = mapping[key].first;
                int dim = mapping[key].second;
                // Replace the key in the name with the new key.
                name = replace_all(name, key, new_key);
                // Distribute the tensor parts across model parallel shards.
                for (int i = 0; i < mp; i++) {
                    torch::Tensor new_param = param;
                    // If the tensor name contains "experts" and does not contain "shared_experts", handle expert splitting.
                    if (name.find("experts") != std::string::npos && name.find("shared_experts") == std::string::npos) {
                        std::vector<std::string> subparts = split_string(name, '.');
                        if (subparts.size() < 3) {
                            throw std::runtime_error("Invalid experts tensor name format: " + name);
                        }
                        // Get the third last element as index.
                        int idx = std::stoi(subparts[subparts.size() - 3]);
                        if (idx < i * n_local_experts || idx >= (i + 1) * n_local_experts) {
                            continue;
                        }
                    }
                    // Else if dim is specified (not -1), narrow the tensor along the specified dimension.
                    else if (dim != -1) {
                        if (param.size(dim) % mp != 0) {
                            throw std::runtime_error("Parameter size along dim " + std::to_string(dim) + " is not divisible by mp.");
                        }
                        int shard_size = (int)param.size(dim) / mp;
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous();
                    }
                    // Save the processed tensor in the state dictionary for the current shard.
                    state_dicts[i][name] = new_param;
                }
            }
            // Free the SafeTensorFile object.
            delete f;
        }
    }
    
    // Create the save_path directory if it does not exist.
    fs::create_directories(save_path);
    
    // Save each model parallel shard into a file.
    for (int i = 0; i < mp; i++) {
        std::string out_file = (fs::path(save_path) / ("model" + std::to_string(i) + "-mp" + std::to_string(mp) + ".safetensors")).string();
        std::cout << "Saving file: " << out_file << std::endl;
        save_file(state_dicts[i], out_file);
    }
    
    // Copy files matching "*token*" from hf_ckpt_path to save_path.
    for (const auto &entry : fs::directory_iterator(hf_ckpt_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            if (file_path.find("token") != std::string::npos) {
                std::string new_file_path = (fs::path(save_path) / entry.path().filename()).string();
                std::cout << "Copying file: " << file_path << " to " << new_file_path << std::endl;
                fs::copy_file(file_path, new_file_path, fs::copy_options::overwrite_existing);
            }
        }
    }
}

Arguments parse_args(int argc, char* argv[]) {
    Arguments args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--hf-ckpt-path" && i + 1 < argc) {
            args.hf_ckpt_path = argv[++i];
        } else if (arg == "--save-path" && i + 1 < argc) {
            args.save_path = argv[++i];
        } else if (arg == "--n-experts" && i + 1 < argc) {
            args.n_experts = std::stoi(argv[++i]);
        } else if (arg == "--model-parallel" && i + 1 < argc) {
            args.model_parallel = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument or missing value: " << arg << std::endl;
            exit(1);
        }
    }
    if (args.hf_ckpt_path.empty() || args.save_path.empty() || args.n_experts <= 0 || args.model_parallel <= 0) {
        std::cerr << "Usage: --hf-ckpt-path <path> --save-path <path> --n-experts <int> --model-parallel <int>" << std::endl;
        exit(1);
    }
    return args;
}

