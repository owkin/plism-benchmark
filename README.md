<div align="center">

![header](https://capsule-render.vercel.app/api?type=waving&height=140&color=gradient&text=Plismbench:&section=header&fontAlign=16&fontSize=45&textBg=false&descAlignY=45&fontAlignY=20&descSize=20&desc=A%20%20robustness%20%20benchmark%20%20of%20%20pathology%20%20foundation%20A%20models&descAlign=52)



[![Python dev](https://github.com/owkin/plism-benchmark/actions/workflows/python-app.yml/badge.svg)](https://github.com/owkin/plism-benchmark/actions/workflows/python-app.yml) [![Deploy doc](https://github.com/owkin/plism-benchmark/actions/workflows/page.yml/badge.svg)](https://github.com/owkin/plism-benchmark/actions/workflows/page.yml) [![Arxiv](https://img.shields.io/badge/Arxiv-2407.18449-red?style=flat-square)](https://arxiv.org/abs/2501.16239)
![GitHub last commit](https://img.shields.io/github/last-commit/owkin/plism-benchmark?style=flat-square)
[![Hugging face](https://img.shields.io/badge/%F0%9F%A4%97%20%20-PLISM-yellow)](https://huggingface.co/datasets/owkin/plism-dataset)
</div>


## Documentation

The documentation can be found [here](https://owkin.github.io/plism-benchmark).
Please refer to the Installation section to install this repository.

## Run PLISM benchmark with your model

The following commands can be run through the cli command `plismbench`.
You can find a detailed description of each subcommand by typing:

```bash
plismbench --help
```

## Hardware requirements

This benchmark can be executed on cpu or gpu. We strongly advise to run it on gpu to benefit from `cupy` acceleration on graphical cards. From downloading to computing the results, running the benchmark takes approximately on our workstation (**32 CPUs, 1 Nvidia T4 (16Go) and 120Gb RAM**):

- 2h45 for a ViT-B: 15 minutes for download, 1h30 for features extraction, 1h for robustness metrics computation.
- 4h45 for a ViT-g: 15 minutes for download, 3h for features extraction, 1h30 for robustness metrics computation.


### Download

First you will need to download [PLISM dataset](https://huggingface.co/datasets/owkin/plism-dataset) hosted on Hugging Face using the following command:

```bash
plismbench download --download-dir /your/download/dir --token your_hf_token --workers 8
```

> [NOTE!]
> 225 Go are required to store 91 WSI-level .h5 files, download approximately takes 10 minutes (32 workers)
>

### Features extraction

Please follow these next steps:

0. Let's set `org=your_company_or_group_name`
1. Implement your model in ``plismbench/models/org.py``
2. Add it to the ``plismbench.models.__init__.py`` enum
3. Add related test in ``tests/models/test_org.py``
4. Perform features extraction using the following script (example with `H0_mini`):

```bash
plismbench extract \
    --extractor h0_mini \
    --batch-size 8 \
    --export-dir /your/features/export/dir/ \
    --download-dir /the/previous/download/dir/ \
    --workers 8
```

> [NOTE!]
> 10 Gb storage and 1h30 are necessary to extract all features with a ViT-B model, 16 CPUs and 1 Nvidia T4 (16Go).
>

> [IMPORTANT!]
> Tests will be run on CI/CD, which implies that private models hosted on Hugging Face will break it:
> ```yaml
>     - name: Log in to Hugging Face
>        run: python -c "from huggingface_hub import login; login(token='${{ secrets.HF_TOKEN }}', new_session=False)"
>```
> We only add public models to this repository.

### Compute metrics

Simply run (example with `H0_mini`):

```bash
plismbench evaluate \
    --extractor h0_mini \
    --features-dir /your/features/previous/export/dir/ \
    --metrics-dir /your/metrics/export/dir/
```

> [NOTE!]
> 1h is necessary to compute metrics for a ViT-B model, 16 CPUs and 1 Nvidia T4 (16Go).
>


Note that the `evaluate` pipeline runs regardless of the models registered in ``plismbench.models.__init__.py``. The only requirement is to store your model features inside `/your/features/previous/export/dir/` under the `extractor` folder (e.g. `your/features/previous/export/dir/h0_mini/`).

The `evaluate` command can run on two different types of device:

- `--device="cpu"` (uses `numpy`): in that case, please specify a number of `--workers`. Metrics computation will be parallelized over all possible slide pairs. **Depending on your RAM, setting a too high number of workers will cause memory errors**. Indeed, if `n_tiles=8139` and `workers=32` then 32 matrices of shape (8139, d) will be stored to RAM, then 32 matrices of shape (16278, 16278) to compute top-k accuracies as it requires to compute dot products between slide A and slide B. Please lower the number of workers if you encounter RAM issues.
- `--device="gpu"` (uses `cupy`): in that case, no need to specify the number of workers. Matrix operations are done on the gpu directly in a sequential manner over all possible slide pairs. **Depending on your GPU RAM, you may encounter cuda memory errors**. We advise to switch to CPU in that case. As an example, we manage to run `evaluate` on GPU (1 T4 16 Go) with Virchow2 concatenated features (d=2563) and `n_tiles=8139` (1 hour).


> [IMPORTANT!]
> The `evaluate` command will compute metrics for each slide-pair (individual pickles and a final .csv with 1 row per pair) and metrics aggregated over pairs (.csv file). Metrics are cosine similarity and top-k accuracies (with k=[1, 3, 5, 10]) by default. We compute mean (std) and median (iqr) over all possible slides pairs, inter-scanners pairs, inter-stainings pairs and inter-scanners + inter-staining pairs.
>
> The number of tiles can be set to either 460 (debugging purposes), 2713 (1/6th of the total number of tiles per slide which is 16278), 5426 (1/3rd), 8139 (half) or 16278 (total number of tiles). **If `None`, the default number of tiles will be 8139 which is the reference for our benchmark.**

### Get your results

By default, results are available at `/your/metrics/export/dir/8139_tiles/your_extractor/results.csv`. Here is an example with `H0_mini` on a subset of 460 tiles. For each type of robustness and metrics, we report `mean (std) ; median (iqr)`.

|                               | cosine_similarity             | top_1_accuracy                | top_3_accuracy                | top_5_accuracy                | top_10_accuracy               |
|:------------------------------|:------------------------------|:------------------------------|:------------------------------|:------------------------------|:------------------------------|
| inter-scanner                 | 0.914 (0.051) ; 0.923 (0.056) | 0.673 (0.255) ; 0.701 (0.433) | 0.781 (0.213) ; 0.835 (0.307) | 0.823 (0.193) ; 0.882 (0.251) | 0.875 (0.162) ; 0.931 (0.173) |
| inter-staining                | 0.769 (0.160) ; 0.830 (0.085) | 0.190 (0.167) ; 0.152 (0.213) | 0.309 (0.218) ; 0.292 (0.307) | 0.372 (0.240) ; 0.374 (0.336) | 0.467 (0.266) ; 0.501 (0.357) |
| inter-scanner, inter-staining | 0.737 (0.156) ; 0.792 (0.106) | 0.104 (0.108) ; 0.072 (0.127) | 0.197 (0.163) ; 0.166 (0.227) | 0.253 (0.190) ; 0.231 (0.274) | 0.346 (0.226) ; 0.346 (0.335) |
| all                           | 0.753 (0.158) ; 0.803 (0.106) | 0.153 (0.194) ; 0.089 (0.166) | 0.251 (0.229) ; 0.195 (0.273) | 0.307 (0.244) ; 0.266 (0.321) | 0.397 (0.265) ; 0.387 (0.373) |

You can generate those results from a `metrics.csv` file by using

```python
from plismbench.utils.aggregate import get_results
results = get_results(pd.read_csv("metrics.csv"), top_k=[1, 3, 5, 10]).
```


## Contribute

Please refer to our [documentation](https://owkin.github.io/plism-benchmark) to follow our contribution guidelines.

> [IMPORTANT!]
> Please report the output of `results.csv` in your PR description as illustrated above, along with the number of tiles used to compute the metrics.
>

## License

This dataset is licensed under [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/deed.en).

## Acknowledgments

We thank PLISM dataset's authors for their unique contribution.

## Third-party licenses

- PLISM dataset (Ochi et al., 2024) is distributed under [CC BY 4.0 license](https://plus.figshare.com/collections/Pathology_Images_of_Scanners_and_Mobilephones_PLISM_Dataset/6773925).
- Elastix (Klein et al., 2010; Shamonin et al., 2014) is distributed under [Apache 2.0 license](https://github.com/SuperElastix/elastix).

## How to cite

If you are using this dataset, please cite the original article (Ochi et al., 2024) and our work as follows:


```
@misc{filiot2025distillingfoundationmodelsrobust,
      title         = {Distilling foundation models for robust and efficient models in digital pathology},
      author        = {Alexandre Filiot and Nicolas Dop and Oussama Tchita and Auriane Riou and Thomas Peeters and Daria Valter and Marin Scalbert and Charlie Saillard and Geneviève Robin and Antoine Olivier},
      year          = {2025},
      eprint        = {2501.16239},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CV},
      url           = {https://arxiv.org/abs/2501.16239},
}
```

## References

- (Ochi et al., 2024) Ochi, M., Komura, D., Onoyama, T. et al. Registered multi-device/staining histology image dataset for domain-agnostic machine learning models. Sci Data 11, 330 (2024).

- (Filiot et al., 2025) Filiot, A., Dop, N., Tchita, O., Riou, A., Peeters, T., Valter, D., Scalbert, M., Saillard, C., Robin, G., & Olivier, A. (2025). Distilling foundation models for robust and efficient models in digital pathology. arXiv. https://arxiv.org/abs/2501.16239

## TODO
- [] Remove tiles sorting
- [] Better comment core_compute_metrics
- [] Add tests for cli commands
- [] Add visualization scripts with different extractors
