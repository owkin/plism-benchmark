extractors=(
    vit_base_pancan_distilled_with_ibot_from_h0_104999
    phikon
    phikon_v2
    virchow2
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
n_tiles=(
    2713
    #5426
    #8364
)

for _n_tiles in ${n_tiles[*]}
do
    plismbench evaluate --extractors "h0_mini" --features-dir /home/owkin/project/plism_features/ --metrics-dir /home/owkin/project/plism_metrics/ --n-tiles $_n_tiles --overwrite
done

for extractor in ${extractors[*]}
do
    cp -r /home/owkin/data/dataset-ddefa0cd-7e56-4a7a-9dac-f24365dff304/histology/processed/features/${extractor} /home/owkin/project/plism_features/
    for _n_tiles in ${n_tiles[*]}
    do
        plismbench evaluate --extractors "${extractor}" --features-dir /home/owkin/project/plism_features/ --metrics-dir /home/owkin/project/plism_metrics/ --n-tiles $_n_tiles
    done
    rm -rf /home/owkin/project/plism_features/${extractor}
done
