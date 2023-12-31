# EDVR (CVPRW'2019)

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://arxiv.org/abs/1905.02716?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529">EDVR (CVPRW'2019)</a></summary>

```bibtex
@InProceedings{wang2019edvr,
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title     = {EDVR: Video restoration with enhanced deformable convolutional networks},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month     = {June},
  year      = {2019},
}
```

</details>

<br/>

Evaluated on RGB channels.
The metrics are `PSNR / SSIM` .

|                                        Method                                         |       REDS4       |                                                                                                                  Download                                                                                                                   |
| :-----------------------------------------------------------------------------------: | :---------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [edvrm_wotsa_x4_8x4_600k_reds](/configs/restorers/edvr/edvrm_wotsa_x4_g8_600k_reds.py) | 30.3430 /  0.8664 | [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522-0570e567.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_wotsa_x4_8x4_600k_reds_20200522_141644.log.json) |
|       [edvrm_x4_8x4_600k_reds](/configs/restorers/edvr/edvrm_x4_g8_600k_reds.py)       | 30.4194 / 0.8684  |       [model](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20210625-e29b71b5.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/edvr/edvrm_x4_8x4_600k_reds_20200622_102544.log.json)       |
