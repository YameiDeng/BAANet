import os


    
    
path_train = 'train'
ftrain = open(os.path.join('./train.txt'), 'w')
total_train = os.listdir(path_train)
for name in total_train:
    p1 = os.path.join(path_train+'/'+name)
    ftrain.write(p1 + ' ')
    p1 = p1.replace('train', 'train_labels').replace('.jpg', '.png')
    ftrain.write(p1 + '\n')

            
ftrain.close()




path_train = 'val'
ftrain = open(os.path.join('./val.txt'), 'w')
total_train = os.listdir(path_train)
for name in total_train:
    p1 = os.path.join(path_train+'/'+name)
    ftrain.write(p1 + ' ')
    p1 = p1.replace('val', 'val_labels').replace('.jpg', '.png')
    ftrain.write(p1 + '\n')

            
ftrain.close()



path_train = 'test'
ftrain = open(os.path.join('./test.txt'), 'w')
total_train = os.listdir(path_train)
for name in total_train:
    p1 = os.path.join(path_train+'/'+name)
    ftrain.write(p1 + ' ')
    p1 = p1.replace('test', 'test_labels').replace('.jpg', '.png')
    ftrain.write(p1 + '\n')

            
ftrain.close()