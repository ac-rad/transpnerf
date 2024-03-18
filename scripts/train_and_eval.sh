#!/bin/bash

#Script to run ns-train and ns-eval on a set of datasets
#Nicole Streltsov
#March 2024

# ./transpnerf/scripts/train_and_eval.sh -m nerfacto
# ./transpnerf/scripts/train_and_eval.sh -m transpnerf

# 1) get the input parameters for the script
while getopts "m" opt; do
    case "$opt" in
        m ) method_name="$OPTARG" ;;
    esac
done

if [ -z "${method_name+x}" ]; then
    echo "Missing method name"
    echo "Usage: $0 -m <method_name>"
fi


echo "------Training and Evaluating Synthetic datasets------"

#SYNTHETIC_DATASETS=("hotdog" "coffee" "wine")
SYNTHETIC_DATASETS=("hotdog")
date
tag=$(date +'%Y-%m-%d')
timestamp=$(date "+%Y-%m-%d_%H%M%S")
trans_file="/transforms.json"
method_opts=()

# configs for later
#--pipeline.model.apply-refl
#--pipeline.model.calc-fresnel
#--pipeline.model.fresnel-version
#--pipeline.datamanager.dataparser.data-id

if [ "$method_name" = "nerfacto" ]; then
    method_opts=(blender-data)
    trans_file=""
fi


cd nerfstudio/scripts/
for dataset in "${SYNTHETIC_DATASETS[@]}"; do

    near_plane=2.0
    far_plane=6.0
    
    if [[ "$dataset" == "wine" ]]; then
        near_plane=6.0
        far_plane=9.0
    fi


    # train
    echo "Launching ${method_name} ${dataset} with timestamp ${timestamp}"
    ns-train "${method_name}" "${method_opts[@]}" \
        --pipeline.model.background-color=white \
        --pipeline.model.disable-scene-contraction=True \
        --pipeline.model.proposal-initial-sampler=uniform \
        --pipeline.model.near-plane=${near_plane} \
        --pipeline.model.far-plane=${far_plane} \
        --experiment-name="synthetic_${dataset}_${tag}" \
        --pipeline.model.use-average-appearance-embedding=False \
        --pipeline.model.distortion-loss-mult=0 \
        --pipeline.model.predict-normals=True \
        --timestamp "$timestamp" \
        --data="data/blender/${dataset}${trans_file}" 
    
    # eval
    wait
    echo "Evaluating ${method_name} ${dataset} with timestamp ${timestamp}"
    config_path="outputs/blender_${dataset}_${tag}/${method_name}/${timestamp}/config.yml"
    ns-eval --load-config="${config_path}" \
            --output-path="output_results/blender_${method_name}_${dataset}_${timestamp}.json"


done
wait
echo "Training and Evaluation Done."


cd ../../
echo "------Creating Excel------"

python3 transpnerf/scripts/get_eval_results.py output_results/ output_evals/final_results.xlsx

wait
echo "Excel creation done."