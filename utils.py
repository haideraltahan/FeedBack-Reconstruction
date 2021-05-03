import copy
import functools
import os
import os.path as osp
import pickle
import re
from multiprocessing import Pool

import lmdb
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import animation as anim
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import stats
from scipy.io import loadmat
from scipy.spatial import distance
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm


def plot_correlations_time(correlations, dir, name='E'):
    threshold = -0.15
    os.makedirs(dir, exist_ok=True)
    max_val = 0
    step_threshold = -0.004
    colors = ["#D6E4FF", "#ADC8FF", "#84A9FF", "#6690FF", "#3366FF", "#254EDB", "#1939B7", "#102693", "#091A7A",
              "#EDFDD8",
              "#D6FBB1", "#B7F489", "#98E969", "#6BDB3B", "#4CBC2B"]
    for idx, layer in enumerate(correlations):
        p_val = layer[1]
        x_val = layer[0]
        all_val = layer[2]
        horizontal_val = np.argwhere(p_val) - 200
        peaks = []
        # if idx > 5:
        #     continue
        color = "#CCCC00" if (name == 'E' and idx == (len(correlations) - 1)) else colors[idx]
        max_val = max(max_val, x_val.max())
        plt.plot(np.arange(-200, 1001, 1), x_val,
                 label="Latent" if (name == 'E' and idx == (len(correlations) - 1)) else f'L{idx + 1}',
                 color=color)
        plt.plot(horizontal_val,
                 np.full_like(horizontal_val, threshold - (step_threshold * idx), np.float),
                 'o',
                 color=color,
                 markersize=0.5)
    plt.ylim(top=max_val + 0.01)
    plt.margins(0)
    plt.legend(loc='upper right', ncol=2)
    plt.savefig(f"{dir}/{name}.png")
    plt.figure()


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def compare_rdms(x, y):
    return np.array(spearmanr(lower_tra(x), lower_tra(y)))[0]


def lower_tra(x):
    return x[np.tril_indices(x.shape[0], -1)]


def get_RDMs(x):
    return loadmat(x)['RDM']


def temp_rdm_compute(meg_rdm, rdm):
    res = []
    meg_rdm = get_RDMs(meg_rdm)
    for i in range(meg_rdm.shape[0]):
        res.append(compare_rdms(meg_rdm[i], rdm))
    return np.array(res)


def twin_ttest(x):
    result = []
    for t in range(x.shape[1]):
        _, p_val = stats.ttest_1samp(x[:, t], 0)
        result.append(p_val)
    return fdrcorrection(np.array(result))[0]


def delete_dir_content(dir):
    if os.path.exists(dir):
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def compare_meg_rdms(meg_rdms, rdm):
    meg_rdms = [os.path.join(meg_rdms, i) for i in os.listdir(meg_rdms)]
    with Pool(13) as p:
        output = p.map(functools.partial(temp_rdm_compute, rdm=rdm), meg_rdms)
    return np.mean(np.array(list(output)), 0), twin_ttest(np.array(list(output))), np.array(list(output))


def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def getActivations(mask, subject_dir):
    mask = np.argwhere(mask)
    s = loadmat(subject_dir)['tma']
    activations = np.zeros((s.shape[0], mask.shape[0]))
    for j in range(s.shape[0]):
        for i in range(mask.shape[0]):
            x, y, z = mask[i]
            activations[j, i] = s[j, x, y, z]
    return activations


def mahalanobis(x, y, IV):
    delta = np.subtract(x, y)
    return np.sqrt(np.dot(np.dot(delta, IV), delta)).item()


def validate(model, val_loader, criterion, Tensor):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.type(Tensor).squeeze()
            target = target.type(torch.cuda.LongTensor).squeeze()
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            pred = torch.nn.functional.softmax(output, 1).argmax(dim=1,
                                                                 keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(val_loader.dataset)
    test_acc = 100. * correct / len(val_loader.dataset)
    return test_loss, test_acc


def mahalanobis_numpy(x, y, IV):
    return np.asarray(
        distance.mahalanobis(np.asnumpy(x), np.asnumpy(y),
                             np.asnumpy(IV)))


def get_data_loader(dataset, batch_size, num_workers):
    """
    generate the data_loader from the given dataset
    :param dataset: F2T dataset
    :param batch_size: batch size of the data
    :param num_workers: num of parallel readers
    :return: dl => dataloader for the dataset
    """
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dl


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)
        return unpacked[0], unpacked[1]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def diff_corr(x, y):
    return 1 - np.corrcoef(x, y)[0, 1]


def euclidean(x, y):
    return np.linalg.norm(x - y)


def computeRDM(data, f=diff_corr):
    matrix = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            d = f(data[i, :], data[j, :])
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix
