from model_vc import Generator
import torch
import torch.nn.functional as F
import time
import datetime
import os

from torch_utils import device

class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.init_model = config.init_model

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.autosave = config.checkpoint_mode=='autosave'
        self.saving_pace = config.save_every_n_iter
        self.saving_prefix = config.save_path

        # Miscellaneous.
        self.device = device
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model()

        if self.init_model:
            self.load_trainable_model(self.init_model)




    def build_model(self):

        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)

        self.G.to(self.device)


    def save_model(self, path = 'autovc.ckpt'):
        torch.save(self.G.state_dict(), path)
        print("model state dict saved at ",path)


    def load_model(self, path = 'autovc.ckpt'):
        if os.path.exists(path):
            print("Load weights from" + path + "for inference")
            self.G.load_state_dict(torch.load(path))
            self.G.eval()
        else:
            print("No checkpoint found, starting from scratch")

    def load_trainable_model(self, path):
        if os.path.exists(self.init_model):
            try:
                checkpoint = torch.load(self.init_model)
                self.G.load_state_dict(checkpoint['G_state_dict'])
                self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            except:
                raise Exception(f'Could not load model at {self.init_model}.')
        else:
            raise Exception(f'Incorrect path: {self.init_model}')

    def save_trainable_model(self, path):
        torch.save({
            'G_state_dict': self.G.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict()
            }, path)


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()


    #=====================================================================================================================================#



    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader

        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']

        # Start training.
        print('Start training...')
        try:
            start_time = time.time()
            for i in range(self.num_iters):

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch data.
                try:
                    x_real, emb_org = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    x_real, emb_org = next(data_iter)

                x_real = x_real.to(self.device)
                x_real_reshaped = x_real.reshape((x_real.shape[0],1,x_real.shape[1],x_real.shape[2]))
                emb_org = emb_org.to(self.device)


                # =================================================================================== #
                #                               2. Train the generator                                #
                # =================================================================================== #

                self.G = self.G.train()

                # Identity mapping loss
                x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
                g_loss_id = F.mse_loss(x_real_reshaped, x_identic)
                g_loss_id_psnt = F.mse_loss(x_real_reshaped, x_identic_psnt)

                # Code semantic loss.
                code_reconst = self.G(x_identic_psnt, emb_org, None)
                g_loss_cd = F.l1_loss(code_real, code_reconst)


                # Backward and optimize.
                g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss = {}
                loss['G/loss_id'] = g_loss_id.item()
                loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
                loss['G/loss_cd'] = g_loss_cd.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training information.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                    for tag in keys:
                        log += ", {}: {:.4f}".format(tag, loss[tag])
                    print(log)

                if self.saving_pace!=0 and (i+1) % self.saving_pace == 0:
                    if not os.path.exists('./trained_models'):
                        os.mkdir('trained_models')
                    self.save_model(f'./trained_models/autovc_{self.saving_prefix}_{i+1}')
        except KeyboardInterrupt:
            if self.autosave:
                self.save_trainable_model('autovc_autosave.ckpt')
                raise Exception('KeyboardInterrupt: autosave done.')
            raise Exception('KeyboardInterrupt: no autosave.')
