
#!/bin/bash

#Script to run ns-train and ns-eval and ns-render on one dataset
#Nicole Streltsov
#March 2024

# to run:
# train_and_eval_each.sh {dataset} {method_name} {timestamp} {tag} {method_opts} {prefix}

# get args
dataset="$1"
method_name="$2"
timestamp="$3"
tag="$4"
prefix="$6"
output_results_folder="$7"

near_plane=2.0
far_plane=6.0
trans_file="/transforms.json"

if [[ "$dataset" == "wine" ]]; then
    near_plane=6.0
    far_plane=9.0
fi

if [ "$method_name" = "nerfacto" ]; then
    trans_file=""
fi

if [[ "$5" == "orig" ]]; then
    method_opts=()
else
    method_opts=()
    IFS=' '
    read -ra method_opts <<< "$5"
fi

cd nerfstudio/scripts/

# train
echo "----- Launching ${method_name} ${dataset} with timestamp ${timestamp} ------"
ns-train "${method_name}" \
    --pipeline.model.background-color white \
    --pipeline.model.disable-scene-contraction=True \
    --pipeline.model.proposal-initial-sampler=uniform \
    --pipeline.model.near-plane "$near_plane" \
    --pipeline.model.far-plane "$far_plane" \
    --experiment-name="${prefix}_${dataset}_${tag}" \
    --pipeline.model.use-average-appearance-embedding=False \
    --pipeline.model.distortion-loss-mult=0 \
    --pipeline.model.predict-normals=True \
    --timestamp "$timestamp" \
    "${method_opts[@]}" \
    --data="../../data/blender/${dataset}${trans_file}" \

wait

# #config_path="outputs/synthetic_hotdog_2024-03-20/nerfacto/2024-03-20_215328/config.yml"
# #output_results="output_results/synthetic_nerfacto_hotdog_2024-03-20_215328"

# config_path="outputs/${prefix}_${dataset}_${tag}/${method_name}/${timestamp}/config.yml"
# output_results="${output_results_folder}/${prefix}_${method_name}_${dataset}_${timestamp}"

# # eval metrics
# echo "------- Evaluating ${method_name} ${dataset} with timestamp ${timestamp} -------"
# ns-eval --load-config="${config_path}" \
#         --output-path="${output_results}.json"
# wait

# # render raw-depth for now
# echo "------- Rendering depth for ${method_name} ${dataset} with timestamp ${timestamp} -------"
# render_output="raw-depth"
# ns-render dataset --load-config="${config_path}" \
#     --rendered-output-names="${render_output}" \
#     --output-path="${output_results}" \
#     --depth-far-plane="${far_plane}" \
#     --depth-near-plane="${near_plane}" \

# wait

# # create point cloud
# ns-export pointcloud \
#     --load-config "${config_path}" \
#     --output-dir "${output_results}" \
#     --num-points 1000000 \
#     --remove-outliers True \
#     --normal-method model_output \
#     --normal-output-name normals \
#     --use-bounding-box True \
#     --bounding-box-min -1.5 -1.5 -1 \
#     --bounding-box-max 1.5 1.5 2

# wait 