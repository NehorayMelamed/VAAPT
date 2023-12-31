# IconVSR (CVPR'2021)

<!-- [ALGORITHM] -->

<details>
<summary align="right">IconVSR (CVPR'2021)</summary>

```bibtex
@InProceedings{chan2021basicvsr,
  author = {Chan, Kelvin CK and Wang, Xintao and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title = {BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond},
  booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition},
  year = {2021}
}
```

</details>

<br/>

Evaluated on RGB channels for REDS4 and Y channel for others. The metrics are `PSNR` / `SSIM` .
The pretrained weights of the IconVSR components can be found here: [SPyNet](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth), [EDVR-M for REDS](https://download.openmmlab.com/mmediting/restorers/iconvsr/edvrm_reds_20210413-3867262f.pth), and [EDVR-M for Vimeo-90K](https://download.openmmlab.com/mmediting/restorers/iconvsr/edvrm_vimeo90k_20210413-e40e99a8.pth).

|                                                            Method                                                           | REDS4 (BIx4)<br>PSNR/SSIM (RGB) | Vimeo-90K-T (BIx4)<br>PSNR/SSIM (Y) | Vid4 (BIx4)<br>PSNR/SSIM (Y) | UDM10 (BDx4)<br>PSNR/SSIM (Y) | Vimeo-90K-T (BDx4)<br>PSNR/SSIM (Y) | Vid4 (BDx4)<br>PSNR/SSIM (Y) |                                                                                                             Download                                                                                                            |
|:---------------------------------------------------------------------------------------------------------------------------:|:-------------------------------:|:-----------------------------------:|:----------------------------:|:-----------------------------:|:-----------------------------------:|:----------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|        [iconvsr_reds4](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/iconvsr/iconvsr_reds.py)       |        **31.6926/0.8951**       |            36.4983/0.9416           |      **27.4809/0.8354**      |         35.3377/0.9471        |            34.4299/0.9287           |        25.2110/0.7732        |       [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413-9e09d621.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_reds4_20210413_222735.log.json)       |
| [iconvsr_vimeo90k_bi](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/iconvsr/iconvsr_vimeo90k_bi.py) |          30.3452/0.8659         |          **37.3729/0.9467**         |        27.4238/0.8297        |         34.2595/0.9398        |            34.5548/0.9295           |        24.6666/0.7491        | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bi_20210413-7c7418dc.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bi_20210413_222757.log.json) |
| [iconvsr_vimeo90k_bd](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/iconvsr/iconvsr_vimeo90k_bd.py) |          29.0150/0.8465         |            34.6780/0.9339           |        26.3109/0.8028        |       **40.0640/0.9697**      |          **37.7573/0.9517**         |      **28.2464/0.8612**      | [model](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bd_20210414-5f38cb34.pth) \| [log](https://download.openmmlab.com/mmediting/restorers/iconvsr/iconvsr_vimeo90k_bd_20210414_084128.log.json) |
