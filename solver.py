from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torchvision import transforms as T
from data_loader import get_loader
import sys
import math

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

    
class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self,config, celeba_args=None, rafd_args=None,celebaHQ_args=None,affectNet_args=None):
        """Initialize configurations."""

        # Data loader.
        self.celeba_args = celeba_args
        self.rafd_args = rafd_args
        self.celebaHQ_args=celebaHQ_args
        self.affectNet_args=affectNet_args

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        # self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lamda_ct = config.lambda_ct

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs
        self.num_steps=int(math.log2(config.image_size)-5) #2^5 = 32
        self.start_step=config.start_step
        
        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            print("Training on device {}".format(self.device))

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_restore_dir = config.model_restore_dir
        self.result_dir = config.result_dir
        self.model_save_dir=config.model_save_dir
        
        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD','CelebA-HQ']:
            self.G = Generator(image_size=self.image_size, c_dim=self.c_dim, repeat_num=self.g_repeat_num)
            self.D = Discriminator(image_size=self.image_size, c_dim=self.c_dim) 
        elif self.dataset in ['Both','HQ']:
            self.G = Generator(image_size=self.image_size,c_dim = self.c_dim+self.c2_dim+2, repeat_num=self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(image_size=self.image_size, c_dim=self.c_dim+self.c2_dim)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        # self.print_network(self.G, 'G')
        # self.print_network(self.D, 'D')
        self.D.to(self.device)
        self.G.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, step,resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_restore_dir, '{}-{}-G.ckpt'.format(step,resume_iters))
        D_path = os.path.join(self.model_restore_dir, '{}-{}-D.ckpt'.format(step,resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
    
    def denorm(self,x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   # grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True)[0]
                                   # only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA' or dataset== 'CelebA-HQ':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA' or dataset == 'CelebA-HQ':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD' or dataset == 'AffectNet':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA' or 'CelebA-HQ':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD' or 'AffectNet':
            return F.cross_entropy(logit, target)
    
    def train(self):
        """Train StarGAN within a single dataset."""
        
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.start_step,self.resume_iters)

        print('Start training...')
        start_time = time.time()
        fade_in=False

        #Conditions for different steps       
        step_iters=[self.num_iters//2]
        for _ in range(1,self.num_steps):
            step_iters.append(self.num_iters)
        step_iters.append(self.num_iters*2)

        for step in range(self.start_step,self.num_steps+1):
            #Change batch size according to image size
            if step in [0,1,2]:
                batch_size=self.batch_size # 32^2, 64^2, 128^2
            elif step in [3,4]:
                batch_size=self.batch_size//2 #256^2, 512^2
            elif step==5:
                batch_size=3 #1024^2

            #Get data_loader based on step
            if self.dataset=='CelebA':
                data_loader=get_loader('CelebA',self.celeba_args,step,batch_size)
            elif self.dataset=='RaFD':
                data_loader=get_loader('RaFD',self.rafd_args,step,batch_size)
            elif self.dataset == 'CelebA-HQ':
                data_loader=get_loader('CelebA-HQ',self.celebaHQ_args,step,batch_size)

            print("Dataset for step {} loaded".format(step))
            
            # get fixed inputs of this step for debugging
            data_iter = iter(data_loader)
            x_fixed, c_org_fixed = next(data_iter)
            x_fixed = x_fixed.to(self.device)
            c_fixed_list = self.create_labels(c_org_fixed, self.c_dim, self.dataset, self.selected_attrs)
           
            # Learning rate cache for decaying.
            g_lr = self.g_lr
            d_lr = self.d_lr
            
            for itr in range(start_iters,step_iters[step]):
            
                # Fade_in only for half the steps when moving on to the next step
                fade_in=(step!=0) and itr<step_iters[step]
                # Weight for fading in only for half the step_iters
                alpha=-1 if not fade_in else min(1,(itr/(step_iters[step]//2))) 
            
                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                #Fetch real images and labels
                try:
                    x_real, label_org = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    x_real, label_org = next(data_iter)
                
                # Generate target domain labels randomly
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if self.dataset == 'CelebA' or self.dataset== 'CelebA-HQ':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                elif self.dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c_dim)
                    c_trg = self.label2onehot(label_trg, self.c_dim)

                x_real = x_real.to(self.device)           # Input images.
                c_org = c_org.to(self.device)             # Original domain labels.
                c_trg = c_trg.to(self.device)             # Target domain labels.
                label_org = label_org.to(self.device)     # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)     # Labels for computing classification loss.
    
                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                requires_grad(self.G,False)
                requires_grad(self.D,True)
                
                # Compute loss with real images.
                out_src_1, out_cls, h_1 = self.D(x_real, step, alpha)
                d_loss_real = -torch.mean(out_src_1)

                #Classification loss
                d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)
                
                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg,step,alpha) #take in step as argument
                out_src, out_cls, _ = self.D(x_fake.detach(),step,alpha) #take in step as argument
                d_loss_fake = torch.mean(out_src)
                
                # Compute loss for gradient penalty.
                eps = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (eps * x_real.data + (1 - eps) * x_fake.data)
                x_hat = Variable(x_hat,requires_grad=True)
                out_src, _, _ = self.D(x_hat,step,alpha) #Take in step as argument
                d_loss_gp = self.gradient_penalty(out_src.sum(), x_hat)
                
                # Compute loss for consistency term
                out_src_, _, h_1_ = self.D(x_real)
                d_CT = torch.mean((out_src_1 - out_src_).view(x_real.size(0), -1), dim = 1)
                d_CT = d_CT.pow(2) + 0.1 * torch.mean((h_1 - h_1_).pow(2), dim = 1)
                d_CT = torch.mean(torch.max(torch.Tensor([0]).to(self.device), d_CT))

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp + self.lamda_ct * d_CT 
                self.reset_grad()
                d_loss.backward(retain_graph=True)
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()
                loss['D/loss_CT'] = d_CT.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #
                
                if (itr+1) % self.n_critic == 0:
                    requires_grad(self.G,True)
                    requires_grad(self.D,False)
                    self.G.zero_grad()
                    self.D.zero_grad()
                    
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg,step,alpha) 
                    out_src, out_cls, _ = self.D(x_fake,step,alpha) 
                    g_loss_fake = - torch.mean(out_src)

                    #Classification loss
                    g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                    # Target-to-original domain.
                    x_reconst= self.G(x_fake, c_org,step,alpha) 
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    # self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()
                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #
                # Print out training information.
                if (itr+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Step {}".format(et, itr+1, step_iters[step],step)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        log_steps=itr+1 if step==0 else step*step_iters[step-1]+itr+1                            
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, log_steps)

                # Translate fixed images for debugging.
                if (itr+1) % self.sample_step == 0:
                    with torch.no_grad():
                        x_fake_list = [x_fixed]
                        for c_fixed in c_fixed_list:
                            x_out=self.G(x_fixed, c_fixed,step,alpha)
                            x_fake_list.append(x_out)
                        x_concat = torch.cat(x_fake_list, dim=3)
                        
                        if self.use_tensorboard:
                            self.logger.image_summary('step-{}'.format(step),x_concat,itr+1)
                        
                        sample_path = os.path.join(self.sample_dir, '{}-{}-images.jpg'.format(step,itr+1))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(sample_path))

                # Save model checkpoints.
                if (itr+1) % self.model_save_step == 0:
                    G_path = os.path.join(self.model_save_dir, '{}-{}-G.ckpt'.format(step,itr+1))
                    D_path = os.path.join(self.model_save_dir, '{}-{}-D.ckpt'.format(step,itr+1))
                    torch.save(self.G.state_dict(), G_path)
                    torch.save(self.D.state_dict(), D_path)
                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                # Decay learning rates.
                if (itr+1) % self.lr_update_step == 0 and (itr+1) > self.num_iters_decay and step==self.num_steps:
                    g_lr -= (self.g_lr / float(self.num_iters_decay))
                    d_lr -= (self.d_lr / float(self.num_iters_decay))
                    self.update_lr(g_lr, d_lr)
                    print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
            start_iters=0
        
    def train_multi_pro(self):
        """Train StarGAN Progressively across multiple datasets"""
        
        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.start_step,self.resume_iters)

        print('Start training...')
        start_time = time.time()
        fade_in=False

        #Conditions for different steps       
        step_iters=[self.num_iters//2]
        for _ in range(1,self.num_steps):
            step_iters.append(self.num_iters)
        step_iters.append(self.num_iters*2)

        for step in range(self.start_step,self.num_steps+1):
            #Change batch size according to image size
            if step in [0,1,2]:
                batch_size=self.batch_size # 32^2, 64^2, 128^2
            elif step in [3,4]:
                batch_size=self.batch_size//2 #256^2, 512^2
            elif step==5:
                batch_size=3 #1024^2

            #Get both data_loaders based on step
            celeba_loader = get_loader('CelebA-HQ',self.celebaHQ_args, step, batch_size)
            rafd_loader = get_loader('RaFD',self.rafd_args, step, batch_size)
            print("Both datasets for step {} loaded".format(step))
            
            # get fixed inputs of this step for debugging
            celeba_iter = iter(celeba_loader)
            rafd_iter = iter(rafd_iter)
            
            x_fixed, c_org_fixed = next(celeba_iter)
            x_fixed = x_fixed.to(self.device)
            
            c_celeba_list = self.create_labels(c_org_fixed, self.c_dim, 'CelebA-HQ', self.selected_attrs)
            c_rafd_list = self.create_labels(c_org_fixed,self.c2_dim,'RaFD')

            zero_celeba = torch.zeros(batch_size,self.c_dim).to(self.device)
            zero_rafd = torch.zeros(batch_size,self.c2_dim).to(self.device)

            mask_celeba = self.label2onehot(torch.zeros(batch_size),2).to(self.device)
            mask_rafd = self.label2onehot(torch.ones(batch_size),2).to(self.device)

            # Learning rate cache for decaying.
            g_lr = self.g_lr
            d_lr = self.d_lr
            
            for itr in range(start_iters,step_iters[step]):
                for dataset in ['RaFD','CelebA-HQ']:

                    # Fade_in only for half the steps when moving on to the next step
                    fade_in = (step!=0) and itr<step_iters[step]
                    
                    # Weight for fading in only for half the step_iters
                    alpha=-1 if not fade_in else min(1,(itr/(step_iters[step]//2))) 
            
                    # ================================================================ #
                    #                             1. Preprocess input data             #
                    # ================================================================ #

                    data_iter = celeba_iter if dataset=='CelebA-HQ' else rafd_iter

                    #Fetch real images and labels
                    try:
                        x_real, label_org = next(data_iter)
                    except:
                        if dataset=='CelebA-HQ':
                            celeba_iter = iter(celeba_loader)
                            x_real, label_org = next(celeba_iter)
                        elif dataset=='RaFD':
                            rafd_iter = iter(rafd_iter)
                            x_real, label_org = next(rafd_iter)

                    # Generate target domain labels randomly
                    rand_idx = torch.randperm(label_org.size(0))
                    label_trg = label_org[rand_idx]

                    x_real = x_real.to(self.device)           # Input images.
                    label_org = label_org.to(self.device)     # Labels for computing classification loss.
                    label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

                    if dataset == 'CelebA-HQ':
                        c_org = label_org.clone()
                        c_trg = label_trg.clone()
                        zero = torch.zeros(x_real.size(0),self.c2_dim).to(self.device)
                        mask = self.label2onehot(torch.zeros(x_real.size(0)),2).to(self.device)
                        c_org = torch.cat([c_org,zero,mask],dim=1)
                        c_trg = torch.cat([c_trg,zero,mask],dim=1)
                    elif dataset == 'RaFD':
                        label_org = self.label2onehot(label_org, self.c2_dim).to(self.device)
                        label_trg = self.label2onehot(label_trg, self.c2_dim).to(self.device)
                        zero = torch.zeros(x_real.size(0),self.c_dim).to(self.device)
                        mask = self.label2onehot(torch.ones(x_real.size(0)),2).to(self.device)
                        c_org = torch.cat([zero, label_org, mask],dim=1)
                        c_trg = torch.cat([zero, label_trg, mask],dim=1)

                    c_org = c_org.to(self.device)             # Original domain labels.
                    c_trg = c_trg.to(self.device)             # Target domain labels.
                    
                    # =================================================================================== #
                    #                             2. Train the discriminator                              #
                    # =================================================================================== #
                    requires_grad(self.G,False)
                    requires_grad(self.D,True)
                    
                    # Compute loss with real images.
                    out_src_1, out_cls, h_1 = self.D(x_real,step,alpha)
                    out_cls = out_cls[:,:self.c_dim] if dataset=='CelebA-HQ' else out_cls[:,self.c_dim:]
                    d_loss_real = -torch.mean(out_src)
                    #Classification loss
                    d_loss_cls = self.classification_loss(out_cls, label_org, dataset=dataset)
                    
                    # Compute loss with fake images.
                    x_fake = self.G(x_real, c_trg,step,alpha) #take in step as argument
                    out_src, out_cls, _ = self.D(x_fake.detach(),step,alpha) #take in step as argument
                    d_loss_fake = torch.mean(out_src)
                    
                    # Compute loss for gradient penalty.
                    eps = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                    x_hat = (eps * x_real.data + (1 - eps) * x_fake.data)
                    x_hat = Variable(x_hat,requires_grad=True)
                    out_src, _, _ = self.D(x_hat,step,alpha) #Take in step as argument
                    d_loss_gp = self.gradient_penalty(out_src.sum(), x_hat)
                    
                    # Compute loss for consistency term
                    out_src_, _, h_1_ = self.D(x_real)
                    d_CT = torch.mean((out_src_1 - out_src_).view(x_real.size(0), -1), dim=1)
                    d_CT = d_CT.pow(2) + 0.1 * torch.mean((h_1 - h_1_).pow(2), dim = 1)
                    d_CT = torch.mean(torch.max(torch.Tensor([0]).to(self.device), d_CT))

                    # Backward and optimize.
                    d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp + self.lamda_ct * d_CT
                    self.reset_grad()
                    d_loss.backward(retain_graph=True)
                    self.d_optimizer.step()

                    # Logging.
                    loss = {}
                    loss['D/loss_real'] = d_loss_real.item()
                    loss['D/loss_fake'] = d_loss_fake.item()
                    loss['D/loss_cls'] = d_loss_cls.item()
                    loss['D/loss_gp'] = d_loss_gp.item()

                    # =================================================================================== #
                    #                               3. Train the generator                                #
                    # =================================================================================== #
                    
                    if (itr+1) % self.n_critic == 0:
                        requires_grad(self.G,True)
                        requires_grad(self.D,False)
                        
                        # Original-to-target domain.
                        x_fake = self.G(x_real, c_trg,step,alpha) 
                        out_src, out_cls = self.D(x_fake,step,alpha) 
                        out_cls=out_cls[:,:self.c_dim] if dataset=='CelebA-HQ' else out_cls[:,self.c_dim:]
                        g_loss_fake = - torch.mean(out_src)

                        #Classification loss
                        g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                        # Target-to-original domain.
                        x_reconst= self.G(x_fake, c_org,step,alpha) 
                        g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                        # Backward and optimize.
                        g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                        self.reset_grad()
                        g_loss.backward()
                        self.g_optimizer.step()

                        # Logging.
                        loss['G/loss_fake'] = g_loss_fake.item()
                        loss['G/loss_rec'] = g_loss_rec.item()
                        loss['G/loss_cls'] = g_loss_cls.item()
                    # =================================================================================== #
                    #                                 4. Miscellaneous                                    #
                    # =================================================================================== #
                    # Print out training information.
                    if (itr+1) % self.log_step == 0:
                        et = time.time() - start_time
                        et = str(datetime.timedelta(seconds=et))[:-7]
                        log = "Elapsed [{}], Iteration [{}/{}], Step {}".format(et, itr+1, step_iters[step],step)
                        for tag, value in loss.items():
                            log += ", {}: {:.4f}".format(tag, value)
                        print(log)

                        if self.use_tensorboard:
                            log_steps=itr+1 if step==0 else step*step_iters[step-1]+itr+1                            
                            for tag, value in loss.items():
                                self.logger.scalar_summary(tag, value, log_steps)

                    # Translate fixed images for debugging.
                    if (itr+1) % self.sample_step == 0:
                        with torch.no_grad():
                            x_fake_list = [x_fixed]
                            
                            for c_fixed in c_celeba_list:
                                c_trg=torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                                x_fake_list.append(self.G(x_fixed, c_trg, step, alpha))

                            for c_fixed in c_rafd_list:
                                c_trg=torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                                x_fake_list.append(self.G(x_fixed, c_trg, step, alpha))
                            x_concat = torch.cat(x_fake_list, dim=3)
                            
                            if self.use_tensorboard:
                                self.logger.image_summary('step-{}'.format(step),x_concat,itr+1)
                            
                            sample_path = os.path.join(self.sample_dir, '{}-{}-images.jpg'.format(step,itr+1))
                            save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                            print('Saved real and fake images into {}...'.format(sample_path))

                    # Save model checkpoints.
                    if (itr+1) % self.model_save_step == 0:
                        G_path = os.path.join(self.model_save_dir, '{}-{}-G.ckpt'.format(step,itr+1))
                        D_path = os.path.join(self.model_save_dir, '{}-{}-D.ckpt'.format(step,itr+1))
                        torch.save(self.G.state_dict(), G_path)
                        torch.save(self.D.state_dict(), D_path)
                        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                    # Decay learning rates.
                    if (itr+1) % self.lr_update_step == 0 and (itr+1) > self.num_iters_decay and step==self.num_steps:
                        g_lr -= (self.g_lr / float(self.num_iters_decay))
                        d_lr -= (self.d_lr / float(self.num_iters_decay))
                        self.update_lr(g_lr, d_lr)
                        print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
            start_iters=0
        
    def train_multi(self):
        """Train StarGAN with multiple datasets."""        
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)           # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']: #Iterating between the two

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                
                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter
                
                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)             # Input images.
                c_org = c_org.to(self.device)               # Original domain labels.
                c_trg = c_trg.to(self.device)               # Target domain labels.
                label_org = label_org.to(self.device)       # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)       # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                #IMPORTANT
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()
            
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self,step=None,max_images=5):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator. 
        self.restore_model(self.test_iters)
        step=self.num_steps if step is None else step
        # Set data loader.
        if self.dataset=='CelebA':
            data_loader=get_loader(self.celeba_args,step,self.batch_size)
        elif self.dataset=='RaFD':
            data_loader=get_loader(self.rafd_loader,step,self.batch_size)

        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                if i>max_images: break
                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                x_fake_list = [x_real]
                _,org_embedding=self.G(x_real,c_org,step=step)#Get original embedding

                #Iterate over target attributes
                interp_path=os.path.join(self.result_dir,'{}-interp'.format(i))
                if not os.path.exists(interp_path):
                    os.makedirs(interp_path)

                for curr_i,c_trg in enumerate(c_trg_list):
                    x_trgt,trgt_embedding=self.G(x_real, c_trg,step=self.num_steps)
                    #Interpolate between target image and src image
                    self.interpolation(org_embedding,trgt_embedding,self.num_steps,interp_path,i,curr_i)
                    x_fake_list.append(x_trgt)
                
                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def interpolation(self,src_latent,trgt_latent,step,interp_path=os.getcwd(),num=5,idx=0,curr_i=0):
        """ Generate 'num' interpolated images b/w src and target"""
        all_imgs=[]
        for i in range(num+1):
            curr_latent=src_latent+ i * (trgt_latent-src_latent)/num #Interpolate!
            with torch.no_grad():
                fake_img=self.G(curr_latent,step=self.num_steps,partial=True)
                all_imgs.append(fake_img)
        all_imgs_concat = torch.cat(all_imgs, dim=3)
        interp_file=os.path.join(interp_path,'{}-{}-interp.jpg'.format(curr_i,idx))
        save_image(self.denorm(all_imgs_concat.data.cpu()),interp_file,nrow=1,padding=0)

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
