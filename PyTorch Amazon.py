
from __future__ import print_function

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, time, sys, copy
from time import gmtime, strftime
from tqdm import tqdm

from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import cv2
from PIL import Image, ImageEnhance

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

## Import self-defined modules ###
import PyTorch_models as my_models
import Image_transformation as IT

import warnings
warnings.filterwarnings("ignore")


#####################################################
train_img_path = '../input/train-jpg/'
test_img_path = '../input/test-jpg/'
img_ext = '.jpg'

model_weight_file = './PyTorch_densenet_v1.hdf5'
pretrained_weight_file = './densenet121_weights_tf.h5'

num_classes = 17
patience = 10

img_dim_1 = 224       # 224 standard input size for densenet, resnet, VGG
img_dim_2 = 299       # 299 input size for inception_v3

num_epochs = 40 #25
batch_size = 20

lr = 1e-4   #1e-3 for SGD; 1e-4 for Adam
lr_decay_epoch = 12

warm_start = True

randomTTA = False
TTA_num_train = 7  # If randomTTA False, automatically set to 7
TTA_num_test = 7   # If randomTTA False, automatically set to 7

sharperness_factor = 1.6
contrast_factor = 1.05

run_training = True
generate_predictions = True
threshold_optimisation = True
make_submission = True

#####################################################
start_time = time.time()
print()
print('Start Time: {}'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

df_train_all = pd.read_csv('../input/train_v2.csv')
df_test = pd.read_csv('../input/sample_submission_v2.csv')

mlb = MultiLabelBinarizer()
mlb.fit_transform( df_train_all['tags'].str.split()).astype(np.float32)

print('-' * 10), print('Labels:'), print('-' * 10)
for label in list(mlb.classes_):
    print(label)
print('-' * 50)


######################################################
def get_train_valid_df(df_train_all, val_size=0.2):
    train_size = 1 - val_size
    K = int(len(df_train_all) * train_size)

    df_train_all = df_train_all.sample(frac=1).reset_index(drop=True)
    df_train = df_train_all[:K].reset_index(drop=True)
    df_valid = df_train_all[K:].reset_index(drop=True)

    return df_train, df_valid


######################################################
repeat_splitting = True
split_count = 0

while repeat_splitting:

    df_train, df_valid = get_train_valid_df(df_train_all, val_size=0.2)

    # Code below make sure the valid set has enough samples for rare labels
    mlb_valid = MultiLabelBinarizer()
    Y = mlb_valid.fit_transform( df_valid['tags'].str.split()).astype(np.float32)
    labels = list(mlb.classes_)
    split_count += 1

    idx1 = labels.index('blow_down')  #Only 98 images in Train set
    idx2 = labels.index('conventional_mine') #Only 100 images in Train set

    a1 = np.sum(Y[:,idx1])
    a2 = np.sum(Y[:,idx2])

    if (len(mlb_valid.classes_) == num_classes) and a1 >= 25 and a2 >= 25:
        print('Train valid split count = {}'.format(split_count))
        print('Valid data: blow_down count = {}; conventional_mine count = {}'.format(a1, a2))
        repeat_splitting = False

######################################################

class KaggleAmazonDataset(Dataset):
    ## From: https://www.kaggle.com/mratsim/starting-kit-for-pytorch-deep-learning
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, df, img_path, img_ext='.jpg', transform=None):

        assert df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all()

        self.mlb = MultiLabelBinarizer()
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = df['image_name']
        self.y_train = self.mlb.fit_transform(df['tags'].str.split()).astype(np.float32)
        self.tags = df['tags']

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)

        img = ImageEnhance.Sharpness(img).enhance( sharperness_factor)
        img = ImageEnhance.Contrast(img).enhance( contrast_factor)

        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label, self.tags[index]

    def __len__(self):
        return len(self.X_train.index)


######################################################
def get_y_train(df=df_train):
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform( df['tags'].str.split()).astype(np.float32)
    #print('Labels: {}'.format(list(mlb.classes_)))
    return Y

#######################################################
def exp_lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_epoch=lr_decay_epoch):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

#######################################################
def fixed_lr_scheduler(optimizer, epoch, init_lr=lr): #lr=0.01
    lr = init_lr
    if epoch >= 6: lr = init_lr * 0.1
    if epoch >= 12: lr = init_lr * 0.01
    if epoch >= 18: lr = init_lr * 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print('LR is set to {}'.format(lr))
    return optimizer

#######################################################
def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    # We need to use average='samples' here, any other average method will generate bogus results
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

######################################################
def train_model(model, optimizer, lr_scheduler, num_epochs=30):

    best_model = model
    best_loss = 1.0
    best_epoch = 0
    patience_count = 0

    for epoch in range(num_epochs):
        since = time.time()


        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(mode=True)   # Set model to training mode
            else:
                model.train(mode=False)  # Set model to evaluate mode

            running_loss = 0.0
            data_count = 0
            predictions = []
            y_true = []

            #Used in inter-epoch print out
            num_print_per_epoch = 10
            num_batches = int(dset_sizes[phase] // batch_size + 1)
            A = int(num_batches // num_print_per_epoch + 1)

            # Iterate over data in batches
            for batch_idx, data in enumerate(dset_loaders[phase]):
                inputs, labels, _ = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                data_count += len(inputs)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                loss = F.binary_cross_entropy(outputs, labels)
                running_loss += batch_size * loss.data[0]

                if (phase == 'train'):
                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                # Print inter-epoch output
                if (batch_idx % A == 0):
                    # print('{} Epoch: {}   [{}/{} ({:.0f}%)]  \tLoss: {:.6f}'.format(
                    #     phase, epoch, batch_idx * len(inputs), dset_sizes[phase],
                    #                   100. * batch_idx / num_batches, loss.data[0]))

                    num_processed = batch_idx * len(inputs)
                    print('{} Epoch: {}   [{}/{} ({:.0f}%)]  \tLoss: {:.6f}'.format(
                        phase, epoch, num_processed, dset_sizes[phase],
                                      100. * num_processed / dset_sizes[phase], loss.data[0]))


                if phase == 'val':
                    output_numpy = outputs.cpu().data.numpy().reshape(-1, num_classes)
                    predictions = np.vstack((predictions, output_numpy)) if batch_idx > 0 else output_numpy

                    labels_numpy = labels.cpu().data.numpy().reshape(-1, num_classes)
                    y_true = np.vstack((y_true, labels_numpy)) if batch_idx > 0 else labels_numpy


            epoch_loss = running_loss / dset_sizes[phase]
            print('{} Loss: {:.4f}'.format( phase, epoch_loss))

            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_model = copy.deepcopy(model)
                    torch.save(best_model, '../modelsnapshot/best_current_model.torch')
                    patience_count = 0
                else:
                    patience_count += 1
                    print('Patience count: {}'.format(patience_count))

            assert (data_count == dset_sizes[phase])

        f_score = fbeta_score(y_true, predictions > 0.2, beta=2, average='samples')
        print('Validation Fbeta_score: {:.6f}'.format(f_score))

        time_elapsed = time.time() - since
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('')

        if patience_count >= patience:
            break

    print('Best result in epoch: {}'.format(best_epoch))
    torch.save(best_model, '../modelsnapshot/best_final_model.torch')

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('')

    return best_model, predictions

#####################################################
def predict(model, dataset_loader, to_evaluate=True):

    since = time.time()

    N = len(dataset_loader.dataset)
    model.train(mode=False)  # Set model to evaluate mode

    # Used in inter-epoch print out
    # num_print_per_epoch = 4
    # num_batches = int(N // batch_size + 1)
    # A = int(num_batches // num_print_per_epoch + 1)

    running_loss = 0.0
    data_count = 0
    predictions = []
    y_true = []

    # Iterate over data in batches
    for batch_idx, data in enumerate(dataset_loader):
        inputs, labels, tags = data
        inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)

        data_count += len(inputs)

        # forward
        outputs = model(inputs)

        output_numpy = outputs.cpu().data.numpy().reshape(-1, num_classes)
        predictions = np.vstack((predictions, output_numpy)) if batch_idx > 0 else output_numpy

        # Print inter-epoch output
        # if (batch_idx % A == 0):
        #     num_processed = batch_idx * len(inputs)
        #     print('[{}/{} ({:.0f}%)]'.format(num_processed, N, 100. * num_processed / N))

        if to_evaluate:
            loss = F.binary_cross_entropy(outputs, labels)
            running_loss += batch_size * loss.data[0]

            labels_numpy = labels.cpu().data.numpy().reshape(-1, num_classes)
            y_true = np.vstack((y_true, labels_numpy)) if batch_idx > 0 else labels_numpy

    assert (data_count == N)
    #print(( np.shape(predictions), np.shape(y_true)))

    if to_evaluate:
        epoch_loss = running_loss / N
        print('Evaluation Loss: {:.4f}'.format(epoch_loss))

        f_score = fbeta_score(y_true, predictions > 0.2, beta=2, average='samples')
        print('Fbeta_score: {:.6f}'.format(f_score))

    time_elapsed = time.time() - since
    print('Evaluation/Prediction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print()

    return predictions


############## Initialising the model  ##############
#use_gpu = torch.cuda.is_available()

# my_Model, model_name = my_models.inception_v3(num_classes = 17, pretrained=True)
# my_Model, model_name = my_models.resnet34(num_classes = 17, pretrained=True)
# my_Model, model_name = my_models.resnet18(num_classes = 17, pretrained=True)
my_Model, model_name = my_models.densenet121(num_classes = 17, pretrained=True)
# my_Model, model_name = my_models.vgg19(num_classes = 17, pretrained=True)

#ignored_params = list(map(id, param_list))
ignored_params = list(map(id, my_Model.sigmoid.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, my_Model.parameters())

# optimizer = optim.Adam(my_Model.parameters(), lr=lr)

# optimizer = optim.SGD([
#         {'params': base_params, 'lr': lr*0.1},
#         {'params': my_Model.sigmoid.parameters()}
#     ], lr=lr, momentum=0.9)

optimizer = optim.Adam([
        {'params': base_params, 'lr': lr*0.1},
        {'params': my_Model.sigmoid.parameters()}
    ], lr=lr)

my_Model = my_Model.cuda()


##############   Setting Data Loaders   ##############
img_dim = img_dim_2 if model_name == 'Inception_v3' else img_dim_1

#Normalise with Image net Mean and Std when using pre-trained models
normMean = [0.485, 0.456, 0.406]
normStd = [0.229, 0.224, 0.225]

train_transform = [ transforms.Lambda(lambda x: IT.RandomResize(x)),
                    transforms.RandomCrop(img_dim),
                    transforms.Lambda(lambda x: IT.transformations(x, np.random.randint(7))),
                    transforms.ToTensor(),
                    transforms.Normalize(normMean, normStd)
                  ]

val_transform = [ transforms.CenterCrop(img_dim),
                  transforms.Lambda(lambda x: IT.transformations(x, np.random.randint(7))),
                  transforms.ToTensor(),
                  transforms.Normalize(normMean, normStd)
                ]

if model_name == 'Inception_v3':
    train_transform.insert(0, transforms.Scale(340))
    val_transform.insert(0, transforms.Scale(340))

data_transforms = {
    'train': transforms.Compose(train_transform),
    'val': transforms.Compose(val_transform),
}

dsets = {
    'train': KaggleAmazonDataset(df_train, train_img_path, img_ext, data_transforms['train']),
    'val': KaggleAmazonDataset(df_valid, train_img_path, img_ext, data_transforms['val']),
}

dset_loaders = {
    x: DataLoader(
        dsets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,   # 1 for CUDA
        pin_memory=True  # CUDA only
    )
    for x in ['train', 'val']
}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
print('dset sizes: {}'.format(dset_sizes))
print('-' * 50)


########  Training models  ########################
if run_training:
    print('######   {} model to run for {} epochs  ######'.format(model_name, num_epochs))
    print('-' * 50)

    if warm_start:
        my_Model = torch.load('../modelsnapshot/best_current_model.torch')
        print('Continuing from best_current_model')

    best_model, _ = train_model(my_Model, optimizer, exp_lr_scheduler, num_epochs=num_epochs)
    #best_model, _ = train_model(my_Model, optimizer, fixed_lr_scheduler, num_epochs=num_epochs)
    print('-' * 50)

########  Getting predictions ######################
if generate_predictions:
    best_model = torch.load('../modelsnapshot/best_final_model.torch')
    # best_model = torch.load('../modelsnapshot/best_{}.torch'.format())

    for phase in ['train', 'test']:
        pred_list = []

        if not randomTTA:
            TTA_num_train = 7
            TTA_num_test = 7

        print('TTA_num_train: {};   TTA_num_test: {}'.format(TTA_num_train, TTA_num_test))

        for i in range(TTA_num_test):
            print('Running {} TTA prediction, iter {}'.format(phase, i))

            data_transforms['augmentation'] = transforms.Compose([
                    #transforms.CenterCrop(img_dim),
                    transforms.Scale(img_dim),
                    transforms.Lambda(lambda x: IT.transformations(x, choice=i)),
                    transforms.ToTensor(),
                    transforms.Normalize(normMean, normStd)
                ])

            if phase == 'train':
                dsets[phase] = KaggleAmazonDataset(df_train_all, train_img_path, img_ext, data_transforms['augmentation'])
                #dsets[phase] = KaggleAmazonDataset(df_valid, train_img_path, img_ext, data_transforms['augmentation'])
                eval_flag = True
            else:
                dsets[phase] = KaggleAmazonDataset(df_test, test_img_path, img_ext, data_transforms['augmentation'])
                eval_flag = False

            dset_loaders[phase] = DataLoader(
                dsets[phase], batch_size=batch_size*4, shuffle=False, num_workers=1, pin_memory=True)

            pred = predict(best_model, dset_loaders[phase], to_evaluate=eval_flag)
            pred_list.append(pred)

            # if phase == 'valid' and i >= TTA_num_train-1:
            #     break

        TTA_predictions = np.mean(pred_list, axis=0)

        prediction_file = '../submission/TTA_{}_pred_{}.npy'.format(phase, model_name)
        np.save(prediction_file, TTA_predictions)

        print('-' * 50)

####### Threshold optimisation ################
if threshold_optimisation:
    print('Optimise Threshold...')

    p_test = np.load('../submission/TTA_test_pred_{}.npy'.format(model_name))
    p_train = np.load('../submission/TTA_train_pred_{}.npy'.format(model_name))
    y_train = get_y_train( df_train_all)
    # p_train = np.load('../submission/TTA_valid_pred_{}.npy'.format(model_name))
    # y_train = get_y_train(df_valid)

    M = len(p_train)
    C = num_classes

    def get_f2_score(y_true, x_pred, x):
        y_pred = np.zeros((M, C))

        for i in range(C):
            y_pred[:, i] = (x_pred[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y_true, y_pred, beta=2, average='samples')
        return score

    base_threshold = [0.2] * num_classes
    base_line_score = get_f2_score(y_train, p_train, base_threshold)
    print('Base line Train data F2 score: {:.6f}'.format(base_line_score))

    #########################
    def optimise_f2_thresholds(y_true, x_pred, resolution, verbose=True):
        # From: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
        label_count = 0

        x = [0.2] * num_classes
        for c in range(C):
            best_threshold = 0
            best_score = 0
            for i in range(resolution):
                i /= float(resolution)
                x[c] = i
                score = get_f2_score(y_true, x_pred, x)
                if score > best_score:
                    best_threshold = i
                    best_score = score
            x[c] = best_threshold
            if verbose:
                print('{}, best threshold {}, f2-score {:.6f}'.format(c, best_threshold, best_score))
        return x

    optimised_threshold = optimise_f2_thresholds(y_train, p_train, 100)
    optimised_score = get_f2_score(y_train, p_train, optimised_threshold)
    print('Best Train data F2 score: {:.6f}'.format(optimised_score))
    print()

###### Prepare submission ######
if make_submission:
    print('Making submission......')
    p_train = np.load('../submission/TTA_train_pred_{}.npy'.format(model_name))
    #p_train = np.load('../submission/TTA_valid_pred_{}.npy'.format(model_name))
    p_test = np.load('../submission/TTA_test_pred_{}.npy'.format(model_name))
    y_train = get_y_train(df_train_all)

    labels = list(mlb.classes_)
    pred_tags = []

    for i in tqdm(range(len(p_test)), miniters=1000):
        a = p_test[i]
        row_labels = []

        for j in range(num_classes):
            if a[j] >= optimised_threshold[j]:
                row_labels = np.append(row_labels, labels[j])
        pred_tags = np.append(pred_tags, [' '.join(row_labels)])

    df_test = pd.read_csv('../input/sample_submission_v2.csv')
    df_test['tags'] = pred_tags
    submission_file = '../submission/submission_{:.4f}.csv'.format(optimised_score)
    df_test.to_csv(submission_file, index=False)
    print('{} saved'.format(submission_file))

    df_test.head()

    print('Process done. Duration: {:.1f} minutes'.format((time.time() - start_time)/60))

#######################################################

# if __name__ == "__main__":
    #train_model()

