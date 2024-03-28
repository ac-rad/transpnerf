#!/bin/bash

#Script to run ns-train and ns-eval and ns-render on a set of datasets
#Nicole Streltsov
#March 2024

# in workspace folder:
# nohup ./transpnerf/scripts/train_and_eval_master.sh synthetic &
# nohup ./transpnerf/scripts/train_and_eval_master.sh real &

# check status: 
    # ps aux | grep ./transpnerf/scripts/train_and_eval_master.sh
    # cat nohup.out - to view logging

start_time=$(date +%s)

# get the input parameters for the script
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset type>"
    exit 1
fi

dataset_type="$1"

date
tag=$(date +'%Y-%m-%d')
timestamp=$(date "+%Y-%m-%d_%H%M%S")
method_opts=()
output_results_folder="output_results_${dataset_type}_V4"
run_nerfacto=0
run_transpnerf=1
run_excel_create=1

if [ "$dataset_type" = "synthetic" ]; then
    DATASETS=("hotdog" "coffee" "wine")
else
    DATASETS=()
fi

echo "------Training and Evaluating Synthetic datasets------"

# 1) NERFACTO
if [ "$run_nerfacto" = "1" ]; then
    echo "Running nerfacto...."
    prefix="${dataset_type}_orig"
    method_name="nerfacto"
    method_opts='blender-data'

    for dataset in "${DATASETS[@]}"; do
        if [ "$dataset" = "wine" ]; then
            # special case: need to adjust the scene_box
            method_name="transpnerf"
            prefix="${dataset_type}_NERFACTO"
            method_opts='--pipeline.model.apply-refl=False'
        fi

        transpnerf/scripts/train_and_eval_each.sh $dataset $method_name $timestamp $tag $method_opts $prefix $output_results_folder
        wait
    done
fi

# 2) TRANSPNERF
if [ "$run_transpnerf" = "1" ]; then
    echo "Running transpnerf...."
    method_name="transpnerf"
    declare -A method_config_dict
    method_config_dict["nerfacto"]="--pipeline.model.apply_refl=False"
    method_config_dict["nerfacto-depth"]="--pipeline.model.apply_refl=False --pipeline.model.apply-depth-supervision=True"
    method_config_dict["orig"]="orig"
    method_config_dict["orig-fresnel"]="--pipeline.model.calc-fresnel=True"
    method_config_dict["orig-depth"]="--pipeline.model.apply_depth_supervision=True"
    method_config_dict["orig-fresnel-depth"]="--pipeline.model.calc-fresnel=True --pipeline.model.apply-depth-supervision=True"

    for dataset in "${DATASETS[@]}"; do
        for key in "${!method_config_dict[@]}"; do
            echo "Prefix: $key, method-opts: ${method_config_dict[$key]}"
            prefix="${dataset_type}_$key"
            method_opts="${method_config_dict[$key]}"

            transpnerf/scripts/train_and_eval_each.sh $dataset $method_name $timestamp $tag "$method_opts" $prefix $output_results_folder
            wait
        done
    done
fi
    

echo "----Training, Evaluation, and Render Done.------"

if [ "$run_excel_create" = "1" ]; then
    echo "--- Install Dependancies for Excel Creation Script ---"
    pip install openpyxl numba

    echo "------Creating Excel------"
    output_results_folder_final="nerfstudio/scripts/${output_results_folder}/"
    final_results_path="output_evals/final_results_${dataset_type}_V4.xlsx"
    python3 transpnerf/scripts/get_eval_results.py "${output_results_folder_final}" "${final_results_path}"

    wait
    echo "Excel created in $final_results_path"
fi

# elapsed time of the script
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"
