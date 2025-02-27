# List of feature extractors to compute metrics on
extractors=(
    virchow2
    h0_mini
    phikon
    phikon_v2
    uni
    uni2h
    hoptimus0
    provgigapath
    hibou_vit_large
    kaiko_vit_base_8
    kaiko_vit_large_14
    plip
    conch
)

# List of number of tiles to compute metrics on: the reference for leaderboard is 8,139 !
n_tiles=(
    8139
    5426
    2713
)

# Set features and metrics directories
features_dir=/home/owkin/project/plism_features/
metrics_dir=/home/owkin/project/plism_metrics/

# Iterate over number of tiles
for _n_tiles in ${n_tiles[*]}
do
    # Iterate over extractors
    for extractor in ${extractors[*]}
    do
        plismbench evaluate \
        --extractor "${extractor}" \
        --features-dir $features_dir \
        --metrics-dir $metrics_dir \
        --n-tiles $_n_tiles \
        --device "gpu"
    done
done
