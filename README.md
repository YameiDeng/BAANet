# BAANet
  This is our research code of "Bone Age Assessment from Articular Surface and Epiphysis in Hand Radiography using Deep Neural Networks"
  
  If you need any help for the code and data, do not hesitate to leave issues in this repository.
****
## Citation
 
```
@article{BAA2023,
journal={submitted},
title={Bone Age Assessment from Articular Surface and Epiphysis using Deep Neural Networks}
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
