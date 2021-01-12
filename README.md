# Style Image Prior
> [Style Generator Inversion for Image Enhancement and Animation](http://www.vision.huji.ac.il/style-image-prior)  
> Aviv Gabbay and Yedid Hoshen

## Inpainting
| ![image](http://www.vision.huji.ac.il/style-image-prior/img/inpainting/imgHQ00040-corrupted.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/inpainting/imgHQ00040-stylegan.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/inpainting/imgHQ00040.png) |
| :---: | :---: | :---: |
| ![image](http://www.vision.huji.ac.il/style-image-prior/img/inpainting/imgHQ00046-corrupted.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/inpainting/imgHQ00046-stylegan.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/inpainting/imgHQ00046.png) |
| Corrupted | Ours | GT |

## Super-Resolution (128x128 to 1024x1024)
| ![image](http://www.vision.huji.ac.il/style-image-prior/img/super-resolution/imgHQ00095-bicubic.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/super-resolution/imgHQ00095-stylegan.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/super-resolution/imgHQ00095.png) |
| :---: | :---: | :---: |
| ![image](http://www.vision.huji.ac.il/style-image-prior/img/super-resolution/imgHQ00044-bicubic.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/super-resolution/imgHQ00044-stylegan.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/super-resolution/imgHQ00044.png) |
| Bicubic | Ours | GT |

## Re-animation: Animating Obama from a video of Trump 
| ![image](http://www.vision.huji.ac.il/style-image-prior/img/reanimation/trump_1.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/reanimation/trump_2.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/reanimation/trump_3.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/reanimation/trump_4.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/reanimation/trump_5.png) |
| :---: | :---: |  :---: | :---: | :---: |
| ![image](http://www.vision.huji.ac.il/style-image-prior/img/reanimation/obama_1.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/reanimation/obama_2.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/reanimation/obama_3.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/reanimation/obama_4.png) | ![image](http://www.vision.huji.ac.il/style-image-prior/img/reanimation/obama_5.png) |

## Usage
### Dependencies
* python >= 3.6
* numpy >= 1.15.4
* tensorflow-gpu >= 1.12.0
* keras >= 2.2.4
* opencv >= 3.4.4
* tqdm >= 4.28.1

### Getting started
1. Clone the official [StyleGAN](https://github.com/NVlabs/stylegan) repository. 
2. Add the local StyleGAN project to PYTHONPATH.

    For bash users:
```
export PYTHONPATH=<path-to-stylegan-project>
```  

### Style Image Prior for Inpainting
Recovering missing parts of given images along with the respective latent codes can be done as follows:
```
inpainting.py --imgs-dir <input-imgs-dir> --masks-dir <output-masks-dir>
    --corruptions-dir <output-corruptions-dir> --restorations-dir <output-restorations-dir>
    --latents-dir <output-latents-dir>
    [--input-img-size INPUT_IMG_HEIGHT INPUT_IMG_WIDTH]
    [--perceptual-img-size EFFECTIVE_IMG_HEIGHT EFFECTIVE_IMG_WIDTH]
    [--mask-size MASK_HEIGHT MASK_WIDTH]
    [--learning-rate LEARNING_RATE]
    [--total-iterations TOTAL_ITERATIONS]
```

### Style Image Prior for Super-Resolution
Performing super-resolution on given images can be done as follows:
```
super_resolution.py --lr-imgs-dir <input-imgs-dir> --hr-imgs-dir <output-imgs-dir>
    --latents-dir <output-latents-dir>
    [--lr-img-size LR_IMG_HEIGHT LR_IMG_WIDTH]
    [--hr-img-size HR_IMG_HEIGHT HR_IMG_WIDTH]
    [--learning-rate LEARNING_RATE]
    [--total-iterations TOTAL_ITERATIONS]
```

**Note:** StyleGAN inversion is very sensitive to the face alignment.
The target face should be aligned exactly as done in the pipeline which CelebA-HQ was created by.
You may use the alignment method implemented here: 
https://github.com/Puzer/stylegan-encoder/blob/master/align_images.py before applying any of the proposed image restoration methods.

## Citing
If you find this project useful for your research, please cite
```
@article{gabbay2019styleimageprior,
  author    = {Aviv Gabbay and Yedid Hoshen},
  title     = {Style Generator Inversion for Image Enhancement and Animation},
  journal   = {arXiv preprint arXiv:1906.11880},
  year      = {2019}
}
```
