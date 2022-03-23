
# SpeechDrivenTongueAnimation
This project animates the tongue and jaw from EMA data through a ML approach.

## Data

The data can be downloaded from this [link](https://drive.google.com/file/d/1AkbLsj41ftc56HNPWAI-Y26-QK4Bqbo9/view?usp=sharing).

## Installation

## Dependencies

Our best model uses Wav2Vec audio features. For this you need to [download the model](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt) from the [Fairseq repository](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md) and place it under the models folder.

## Pipeline

Our pipeline consists of the following stage:
1. Extract audio features from wav2vec model
2. Build the dataset to train the model
3. Train the landmark prediction model
4. Evaluate the model
5. Visualize the model

### 1. Audio Feature Extraction

### 2. Building the dataset

### 3. Training the model

### 4. Testing the model

### 5. Visualizing the results

## Citation

If you find this work useful on your research, please cite our work:
```
@inproceedings{medina2022speechtongue,
  title={Speech Driven Tongue Animation},
  author={Medina, Salvador and Tomé, Denis and Stoll, Carsten and Tiede, Mark and Munhall, Kevin and Hauptmann, Alex and Matthews, Iain},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022},
  organization={IEEE/CVF}
}
```
