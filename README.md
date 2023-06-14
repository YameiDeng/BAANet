# BAANet
  This is our research code of "Bone Age Assessment from Articular Surface and Epiphysis in Hand Radiography using Deep Neural Networks"
  
  If you need any help for the code and data, do not hesitate to leave issues in this repository.
****
## Citation
 
```
@article{deng2023bone,
  title={Bone age assessment from articular surface and epiphysis using deep neural networks},
  author={Deng, Yamei and Chen, Yonglu and He, Qian and Wang, Xu and Liao, Yong and Liu, Jue and Liu, Zhaoran and Huang, Jianwei and Song, Ting},
  journal={Mathematical Biosciences and Engineering},
  volume={20},
  number={7},
  pages={13133--13148},
  year={2023}
}

```
## Method
### Training
```

python train.py

```

### Testing

```

python test.py

```

## Notes

```

If you want to train the deep leanring networks on the proposed articular surface ("data_a") and epiphysis ("data_e") datasets, please update the path in the train.py:

For the articular surface dataset:
''''
train_dataset_path = './data_e/train'
val_dataset_path = './data_e/val'
test_dataset_path = './data_e/test'
train_csv_path = './data_e/train.csv'
val_csv_path = './data_e/val.csv'
test_csv_path = './data_e/test.csv'
''''


For the epiphysis dataset:
''''
train_dataset_path = './data_a/train'
val_dataset_path = './data_a/val'
test_dataset_path = './data_a/test'
train_csv_path = './data_a/train.csv'
val_csv_path = './data_a/val.csv'
test_csv_path = './data_a/test.csv'
''''

```
