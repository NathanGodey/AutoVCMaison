import argparse
from model_vc import Generator
import matplotlib.pyplot as plt
import torch
from torch_utils import device
from data_loader_circular import get_loader

def show_melsp(tensor, title):
    fig = plt.figure(title)
    plt.imshow(tensor.detach().cpu().reshape((128,80)).numpy().T, cmap='viridis')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str, default='voxceleb')
    parser.add_argument('--random', type=int, default=1)

    config = parser.parse_args()

    checkpoint = torch.load(config.model, map_location=device)

    neck_dim = checkpoint['G_state_dict']['encoder.lstm.weight_hh_l0'].shape[1]
    G = Generator(neck_dim, 256, 512, 16)
    G.load_state_dict(checkpoint['G_state_dict'])
    G.to(device)
    loss = checkpoint["G_loss"]

    dataloader = get_loader(config.dataset + '/spmel', 1, 128)

    data_iter = iter(dataloader)
    x_real, emb_org, emb_target = next(data_iter)

    x_real = x_real.to(device)

    emb_org = emb_org.to(device)

    emb_target = emb_target.to(device)

    # Circular mapping loss
    x_target_pred, x_target_pred_psnt, code_org = G(x_real, emb_org, emb_target)
    x_org_reconst, x_org_reconst_psnt, code_target_pred = G(x_target_pred.reshape(x_real.shape), emb_target, emb_org)

    x_self_recon, x_self_recon_psnt, code_org = (x_real, emb_org, emb_org)
    plt.semilogy(loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

    show_melsp(x_real, 'Real utterance (A)')
    show_melsp(x_target_pred, 'Converted utterance (B)')
    plt.show()

    show_melsp(x_real, 'Real utterance (A)')
    show_melsp(x_target_pred, 'Self-converted utterance (A)')
    plt.show()

    show_melsp(x_target_pred, 'Converted utterance (B)')
    show_melsp(x_target_pred_psnt, 'Converted utterance after PostNet (B)')
    plt.show()

    show_melsp(x_org_reconst, 'Reconstructed utterance (A)')
    show_melsp(x_org_reconst_psnt, 'Reconstructed utterance after PostNet (A)')
    plt.show()

    show_melsp(x_real, 'Real utterance (A)')
    show_melsp(x_org_reconst_psnt, 'Reconstructed utterance after PostNet (A)')
    plt.show()
