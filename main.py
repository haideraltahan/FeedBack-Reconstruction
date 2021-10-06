import itertools
import random
from functools import partial

import fire
import pandas as pd
import seaborn as sns
import torchvision.transforms as transforms
from matplotlib import ticker
from scipy.io import savemat
from scipy.ndimage import zoom
from scipy.stats import sem
from sklearn import manifold
from sklearn.manifold import MDS
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import datasets
from torchvision.utils import save_image, make_grid

from models import *
from utils import *


class Main(object):
    def __init__(self, img_size=224, cuda=torch.cuda.is_available(), images_folder='./data',
                 models_folder='./models', model='aae_dprior', test_image='156imagesC', latent_dim=100,
                 samples_folder='./samples', logs_folder='./logs', n_epochs=2000, batch_size=1024, n_model=None,
                 intermediate_size=512,
                 sample_interval=50, n_cpu=8, n_classes=4, train_data='train_Dataset.lmdb', dist_prior=True,
                 decoder_prior=False, deconv=True, init=True, m=7, random=False):

        super(Main, self).__init__()
        self.init = init
        self.deconv = deconv
        self.dist_prior = dist_prior
        self.decoder_prior = decoder_prior
        self.n_cpu = n_cpu
        self.intermediate_size = intermediate_size
        self.sample_interval = sample_interval
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.n_classes = n_classes
        self.cuda = cuda
        self.m = m
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.batch_size = batch_size * torch.cuda.device_count()
        self.n_epochs = n_epochs
        self.main_image_folder = images_folder
        self.train_images = os.path.join(images_folder, train_data)
        self.images_folder = os.path.join(images_folder, 'test', test_image)
        self.meg_folder = os.path.join(images_folder, 'MEG_RDMs')
        self.main_model_folder = models_folder
        self.models_folder = os.path.join(models_folder, model)
        self.logs_folder = os.path.join(logs_folder, model)
        os.makedirs(self.logs_folder, exist_ok=True)
        self.main_samples_folder = samples_folder
        self.samples_folder = os.path.join(samples_folder, model)
        self.model = model
        self.latent_dim = latent_dim
        self.define_models()
        if cuda:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
            self.discriminator = nn.DataParallel(self.discriminator)
            self.encoder.cuda()
            self.decoder.cuda()
            self.discriminator.cuda()
            summary(self.encoder, input_size=(3, img_size, img_size))
            if decoder_prior:
                summary(self.decoder, input_size=(1, latent_dim + n_classes))
            else:
                summary(self.decoder, input_size=(1, latent_dim))
            if dist_prior:
                summary(self.discriminator, input_size=(1, latent_dim + n_classes))
            else:
                summary(self.discriminator, input_size=(1, latent_dim))
        if not random:
            self.load_model(n_model=n_model, train=False)
        self.writer = SummaryWriter(log_dir=self.logs_folder)

        print(f"tensorboard --logdir={self.logs_folder}")

    def define_models(self):
        if 'aae' in self.model:
            self.encoder = Encoder(self.Tensor, latent_dim=self.latent_dim,
                                   intermediate_size=self.intermediate_size, init=self.init, reparm=True)
        elif 'vae' in self.model:
            self.encoder = Encoder(self.Tensor, latent_dim=self.latent_dim,
                                   intermediate_size=self.intermediate_size, init=self.init, reparm=True,
                                   return_mu=True)
        else:
            self.encoder = Encoder(self.Tensor, latent_dim=self.latent_dim,
                                   intermediate_size=self.intermediate_size, init=self.init, reparm=False)
        self.decoder = Decoder(latent_dim=self.latent_dim, prior=self.decoder_prior, deconv=self.deconv,
                               init=self.init,
                               m=self.m)
        self.discriminator = Discriminator(latent_dim=self.latent_dim, n_classes=self.n_classes,
                                           prior=self.dist_prior,
                                           init=self.init)

    def read_data(self):
        assert os.path.exists(self.logs_folder)
        event_acc = EventAccumulator(self.logs_folder)
        event_acc.Reload()
        print(event_acc.Tags())
        print(event_acc.Scalars('Loss/g_loss'))

    def gen_rdm_plots(self, scatter_size=200, extension='svg'):
        assert os.path.exists(os.path.join(self.samples_folder, 'rdms'))
        save_path = os.path.join(self.samples_folder, 'rdm_plots')
        os.makedirs(save_path, exist_ok=True)
        self.delete_dir_content(save_path)
        for i in os.listdir(os.path.join(self.samples_folder, 'rdms')):
            rdm = normalize(get_RDMs(os.path.join(self.samples_folder, 'rdms', i)))
            plt.figure()
            plt.imshow(rdm)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'RDM_{i.replace(".mat", "")}.{extension}'), bbox_inches='tight',
                        pad_inches=0)
            plt.close()
            plt.figure()
            Y = MDS(n_components=2, n_init=100, max_iter=1000, dissimilarity='precomputed', eps=1e-6)
            Y = Y.fit_transform(rdm)
            plt.figure()
            plt.axis('off')
            plt.scatter(Y[0:32, 0], Y[0:32, 1], c='red', s=scatter_size)
            plt.scatter(Y[32:84, 0], Y[32:84, 1], c='orange', s=scatter_size)
            plt.scatter(Y[84:120, 0], Y[84:120, 1], c='blue', s=scatter_size)
            plt.scatter(Y[120:157, 0], Y[120:157, 1], c='green', s=scatter_size)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'MDS_{i.replace(".mat", "")}.{extension}'), bbox_inches='tight',
                        pad_inches=0)
            plt.close()

    def gen_rdm(self):
        assert os.path.exists(self.images_folder)
        save_path = os.path.join(self.samples_folder, 'rdms')
        os.makedirs(save_path, exist_ok=True)
        print(save_path)
        self.delete_dir_content(save_path)
        dataloader = self.load_data(self.images_folder, 156)

        def hook_fn(m, i, o, e=True):
            o = o.flatten(1).cpu().detach().numpy()
            if e:
                if m in etem_act:
                    etem_act[m] = np.vstack((etem_act[m], o))
                else:
                    etem_act[m] = o
            else:
                if m in dtem_act:
                    dtem_act[m] = np.vstack((dtem_act[m], o))
                else:
                    dtem_act[m] = o

        def get_all_layers(net, par=True):
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose2d):
                    print(layer)
                    layer.register_forward_hook(partial(hook_fn, e=par))

        get_all_layers(self.encoder)
        get_all_layers(self.decoder, False)

        for idx, (img, labels) in enumerate(dataloader):
            etem_act = {}
            dtem_act = {}
            img = img.cuda()
            de_img = self.encoder(self.Tensor(img))
            # print(len(de_img))
            if len(de_img) == 3:
                _ = self.decoder(de_img[0])
                # print(de_img[0].flatten(1).cpu().detach().numpy().shape)
                etem_act['z'] = de_img[0].flatten(1).cpu().detach().numpy()

            else:
                _ = self.decoder(de_img)
                # print(de_img.flatten(1).cpu().detach().numpy().shape)
                etem_act['z'] = de_img.flatten(1).cpu().detach().numpy()

        for i, activations in tqdm(enumerate(etem_act.values())):
            savemat(os.path.join(save_path, f'E_L{i + 1}.mat'),
                    {'RDM': computeRDM(activations.squeeze())})
        for i, activations in tqdm(enumerate(dtem_act.values())):
            savemat(os.path.join(save_path, f'D_L{i + 1}.mat'),
                    {'RDM': computeRDM(activations.squeeze())})

    def gen_compare_rdms(self):
        assert os.path.exists(os.path.join(self.samples_folder, 'rdms'))
        save_path = os.path.join(self.samples_folder, 'compare_rdms')
        os.makedirs(save_path, exist_ok=True)
        self.delete_dir_content(save_path)
        files = os.listdir(os.path.join(self.samples_folder, 'rdms'))
        files.sort()
        files = [os.path.join(self.samples_folder, 'rdms', x) for x in files]
        RDMs = list(map(get_RDMs, files))
        x_labels = [file.split('/')[-1].replace('.mat', '').replace('_L', '') for file in files]
        res = np.zeros((len(files), len(files)))
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                tres = 1 - np.corrcoef(lower_tra(RDMs[i]), lower_tra(RDMs[j]))[0, 1]
                res[i, j] = tres
                res[j, i] = tres

        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(111)
        cax = ax.matshow(res / np.max(res), interpolation='nearest')
        fig.colorbar(cax)

        files = list(map(lambda x: x.split("\\")[-1].replace('.mat', '').split("/")[-1], files))
        ax.set_xticklabels([''] + x_labels)
        ax.set_yticklabels([''] + x_labels)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
                 rotation_mode="anchor")
        plt.savefig(os.path.join(save_path, f'{self.model}.png'))
        savemat(os.path.join(save_path, f'{self.model}.mat'), {'Matrix': res})

    def plot_meg_compare(self):
        assert os.path.exists(self.meg_folder)
        assert os.path.exists(os.path.join(self.samples_folder, 'rdms'))
        self.encoder.eval()
        self.decoder.eval()
        save_path = os.path.join(self.samples_folder, 'rdm_meg')
        os.makedirs(save_path, exist_ok=True)
        # self.delete_dir_content(save_path)
        rdms_folder = os.path.join(self.samples_folder, 'rdms')
        E_correlations = []
        D_correlations = []
        files = [os.path.join(rdms_folder, i) for i in os.listdir(rdms_folder)]
        E_files = [file for file in files if 'E_' in file]
        D_files = [file for file in files if 'D_' in file]
        if os.path.exists(os.path.join(save_path, 'E.pickle')):
            E_correlations = pickle.load(open(os.path.join(save_path, 'E.pickle'), 'rb'))
        else:
            for idx, layer_rdm in tqdm(enumerate(E_files)):
                E_correlations.append(compare_meg_rdms(self.meg_folder, get_RDMs(layer_rdm)))
            pickle.dump(E_correlations, open(os.path.join(save_path, 'E.pickle'), "wb"))
        plot_correlations_time(E_correlations, save_path)
        if os.path.exists(os.path.join(save_path, 'D.pickle')):
            D_correlations = pickle.load(open(os.path.join(save_path, 'D.pickle'), 'rb'))
        else:
            for idx, layer_rdm in tqdm(enumerate(D_files)):
                D_correlations.append(compare_meg_rdms(self.meg_folder, get_RDMs(layer_rdm)))
            pickle.dump(D_correlations, open(os.path.join(save_path, 'D.pickle'), "wb"))
        plot_correlations_time(D_correlations, save_path, 'D')

    def gen_images(self, n_row=8):
        assert os.path.exists(self.images_folder)
        save_path = os.path.join(self.samples_folder, 'image_reconstruction')
        os.makedirs(save_path, exist_ok=True)
        self.delete_dir_content(save_path)
        dataloader = self.load_data(self.images_folder, batch_size=1)
        for idx, (img, _) in enumerate(dataloader):
            decode = self.encoder(img)
            gen_imgs = self.decoder(decode)
            break
        save_image(gen_imgs, os.path.join(save_path, 'f.png'), nrow=n_row, normalize=True)
        save_image(img, os.path.join(save_path, 'r.png'), nrow=n_row, normalize=True)

    def load_data(self, dir, batch_size=None, shuffle=False, lmdb=False):
        assert os.path.exists(dir)
        if not lmdb:
            imageset = datasets.ImageFolder(root=dir, transform=self.img_transform)
        else:
            imageset = ImageFolderLMDB(dir)
        return torch.utils.data.DataLoader(imageset,
                                           batch_size=batch_size if batch_size else self.batch_size,
                                           shuffle=shuffle,
                                           num_workers=self.n_cpu)

    def load_model(self, train=False, n_model=None):
        if train:
            self.discriminator.train()
            self.decoder.train()
            self.encoder.train()
        else:
            self.discriminator.eval()
            self.decoder.eval()
            self.encoder.eval()
        if os.path.exists(self.models_folder) and len(os.listdir(self.models_folder)) > 0:
            files = os.listdir(self.models_folder)
            files.sort(key=natural_keys)
            print(f'Loading Model {files[-1]}....')
            if n_model:
                checkpoint = torch.load(os.path.join(self.models_folder, files[n_model]),
                                        map_location='cuda' if self.cuda else 'cpu')
            else:
                checkpoint = torch.load(os.path.join(self.models_folder, files[-1]),
                                        map_location='cuda' if self.cuda else 'cpu')

            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=True)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=True)

            return int(files[-1])
        print('Not loading a model')
        return 0

    def change_range(self, x):
        return (x + 1) / 2

    def get_latent_disc(self, n_images=15, p=0.00009):
        results = [[] for _ in range(self.n_classes)]
        for j in tqdm(range(self.n_classes)):
            while len(results[j]) < n_images:
                z = Variable(self.Tensor(np.random.normal(0, 1, (n_images, self.latent_dim))))
                labels = nn.functional.one_hot(
                    Variable(self.Tensor(np.array([np.full(n_images, j)]).flatten())).to(torch.int64), self.n_classes)
                zn = torch.cat((labels.float(), z), 1)
                r = self.discriminator(zn)
                for i in range(r.shape[0]):
                    if r[i] <= p:
                        results[j].append(self.decoder(z[i, :].view(1, -1))[0].squeeze().cpu().detach().numpy())
        return results

    def gen_raw_images(self, n_row=5):
        save_path = os.path.join(self.samples_folder, 'gen_images')
        self.delete_dir_content(save_path)
        os.makedirs(save_path, exist_ok=True)
        images = self.get_latent_disc()
        for i, imgs in tqdm(enumerate(images)):
            save_image(torch.tensor(imgs).data, os.path.join(save_path, f'{i}.png'), nrow=n_row, normalize=True)

    def sample_images(self, step, n_row=5, recon=None):
        save_path = os.path.join(self.samples_folder, 'sampled_images')
        os.makedirs(save_path, exist_ok=True)
        z = Variable(self.Tensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim))))
        if self.decoder_prior:
            abels = Variable(
                self.Tensor(np.random.randint(0, self.n_classes, (n_row ** 2))).to(torch.int64))
            z = torch.cat((nn.functional.one_hot(abels, self.n_classes).float().cuda(), z), 1)
        gen_imgs = self.decoder(z)
        # self.save_images(gen_imgs.data, os.path.join(save_path, image_name))
        self.writer.add_image('generated', make_grid(self.change_range(gen_imgs)), step)
        if recon:
            self.writer.add_image('real_images', make_grid(self.change_range(recon[0][:20])), step)
            self.writer.add_image('reconstructed_images', make_grid(self.change_range(recon[1][:20])), step)

    def delete_dir_content(self, dir):
        if os.path.exists(dir):
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))

    def reset_grad(self):
        self.encoder.zero_grad()
        self.discriminator.zero_grad()
        self.decoder.zero_grad()

    def train_vae(self, lr=0.0002, save_model=10):
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr
        )
        Tensor = self.Tensor
        loss_log = tqdm(total=0, position=0, bar_format='{desc}')
        print("Loading data ... ")
        dataloader = self.load_data(self.train_images, shuffle=True, lmdb=True)
        offset = self.load_model(train=True)
        print("Starting the training process ... ")
        for epoch in range(self.n_epochs):
            for i, (imgs, labels) in enumerate(dataloader):
                real_imgs = Variable(imgs.type(Tensor).squeeze())

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                encoded_imgs, mu, log_var = self.encoder(real_imgs)
                decoded_imgs = self.decoder(encoded_imgs)

                # Loss measures generator's ability to fool the discriminator
                g_loss = F.mse_loss(decoded_imgs, real_imgs) + torch.mean(
                    -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

                g_loss.backward()
                optimizer_G.step()

                loss_log.set_description_str(
                    '[Epoch %d/%d] [Batch %d/%d] [G loss: %f]' % (
                        epoch + offset, self.n_epochs, i, len(dataloader), g_loss.item())
                )
                batches_done = (epoch + offset) * len(dataloader) + i
                self.writer.add_scalar('Loss/g_loss', g_loss.item(), batches_done)
                if batches_done % self.sample_interval == 0:
                    self.sample_images(step=batches_done, recon=(real_imgs, decoded_imgs))
            if (epoch + offset + 1) % save_model == 0:
                self.save_model(epoch + offset + 1)

    def train_aae(self, lr=0.0002, save_model=10):
        adversarial_loss = torch.nn.BCELoss()
        pixelwise_loss = torch.nn.MSELoss()
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr
        )
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        Tensor = self.Tensor
        loss_log = tqdm(total=0, position=0, bar_format='{desc}')
        print("Loading data ... ")
        dataloader = self.load_data(self.train_images, shuffle=True, lmdb=True)
        offset = self.load_model(train=True)
        print("Starting the training process ... ")
        for epoch in range(self.n_epochs):
            for i, (imgs, labels) in enumerate(dataloader):
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
                real_imgs = Variable(imgs.type(Tensor).squeeze(), requires_grad=False)
                real_labels = labels.type(torch.cuda.LongTensor).squeeze()

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                encoded_imgs = self.encoder(real_imgs)
                decoded_imgs = self.decoder(encoded_imgs)

                if self.dist_prior:
                    encoded_imgs = torch.cat(
                        (nn.functional.one_hot(real_labels, self.n_classes).float(), encoded_imgs), 1)

                encoded_imgs = Variable(encoded_imgs, requires_grad=True)
                # print(torch.argmax(encoded_imgs), encoded_imgs.max())
                # Loss measures generator's ability to fool the discriminator
                g_loss = 0.001 * adversarial_loss(self.discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
                    decoded_imgs, real_imgs
                )

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise as discriminator ground truth
                z = self.Tensor(imgs.shape[0], self.latent_dim).normal_()

                if self.dist_prior:
                    z = torch.cat(
                        (nn.functional.one_hot(torch.randint(0, self.n_classes, size=[imgs.shape[0]], device='cuda'),
                                               self.n_classes).float(), z), 1)
                z = Variable(z, requires_grad=True)
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(self.discriminator(z), valid)
                fake_loss = adversarial_loss(self.discriminator(encoded_imgs.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                optimizer_D.step()

                loss_log.set_description_str(
                    '[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [D loss: %f]' % (
                        epoch + offset, self.n_epochs, i, len(dataloader), g_loss.item(), d_loss.item())
                )
                batches_done = (epoch + offset) * len(dataloader) + i
                self.writer.add_scalar('Loss/g_loss', g_loss.item(), batches_done)
                self.writer.add_scalar('Loss/d_loss', d_loss.item(), batches_done)
                if batches_done % self.sample_interval == 0:
                    self.sample_images(step=batches_done, recon=(real_imgs, decoded_imgs))
            if (epoch + offset + 1) % save_model == 0:
                self.save_model(epoch + offset + 1)

    def save_images(self, data, image_name, n_row=8):
        save_image(data, image_name, nrow=n_row,
                   normalize=True)

    def save_model(self, epoch):
        os.makedirs(self.models_folder, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
        }, os.path.join(self.models_folder, f"{epoch}"))


if __name__ == '__main__':
    fire.Fire(Main)
