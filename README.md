# TTNet-Pytorch
The implementation for the paper "TTNet: Real-time temporal and spatial video analysis of table tennis"

---

## Source code structure
```shell script
${ROOT}
├──dataset/
    ├──test/
        ├──annotations/
        ├──images/
        ├──videos/
    ├──training
        ├──annotations/
            ├──game_1/
                ├──segmentation_masks/
                ├──ball_markup.json
                ├──events_markup.json
        ├──images/
            ├──game_1/
                ├──img_000001.jpg
                ├──img_000002.jpg
                ...
        ├──videos/
            ├──game_1.mp4
            ├──game_2.mp4
            ...
├──src/
    ├──config/
        ├──config.py
    ├──data_process/
        ├──transformation.py
        ├──ttnet_data_utils.py
        ├──ttnet_dataloader.py
        ├──ttnet_dataset.py
    ├──losses/
        ├──losses.py
    ├──models/
        ├──TTNet.py
    ├──prepare_dataset/
        ├──download_dataset.py
        ├──extract_images.py
        ├──unzip.py
    ├──training/
        ├──main.py
        ├──train_utils.py
    ├──utils/
        ├──logger.py
        ├──misc.py


    
├──README.md
```

## Requirements
You can refer [the tutorial](https://github.com/maudzung/virtual_environment_python3) to install a virtual environment

## Prepare the dataset
### Download the dataset and unzip annotations
```shell script
cd src/prepare_dataset
python download_dataset.py
python unzip.py
```

### Extract images from videos
If you want to extract all images from videos, let's execute:
```shell script
python extract_all_images.py
```

As the authors mentioned in the original paper, images are extracted based on frames that has events. 
To extract selected frames, let's execute:
```shell script
python extract_selected_images.py
```

The source code will be updated soon... Staring and watching us...