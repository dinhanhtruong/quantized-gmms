from torch import nn
import torch
from vector_quantize_pytorch import VectorQuantize
import numpy as np

class VQAutoEncoder(nn.Module):
    def __init__(self, codebook_sz, input_feat_dim, hidden_dim, vq_feat_dim):
        super(VQAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vq_feat_dim),
        )
        self.vq_layer = VectorQuantize(
            dim = vq_feat_dim,
            codebook_size = codebook_sz,     # codebook size
            decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight = 1.   # the weight on the commitment loss
        )

        self.decoder = nn.Sequential(
            nn.Linear(vq_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_feat_dim),
        )

    def forward(self, feats):
        '''
        Args
            - feats: [B,D]

        Returns
            - reconstructed_feats: [B,D]
            - code indices: [B,]
            - commit loss
        '''
        latent_feats = self.encoder(feats)
        quantized_feats, indices, commit_loss = self.vq_layer(latent_feats) # (B, D), (B, ),
        # TODO reshape?
        # temp compare quantized feat vs indexing into raw codebook
        codebook = self.get_codebook()
        return self.decoder(quantized_feats), indices, commit_loss
    
    def decode_from_quantized_latent(self, quantized_latent):
        return self.decoder(quantized_latent)

    def get_codebook(self):
        '''
        Returns
            - [vocab, D]: raw codebook tensor
        '''
        return self.vq_layer._codebook.embed.squeeze()




# import all continuous codes

sample_dir_name = 'raw_gmm_surface_feat_quantize_learned_1024_adaptive'
surface_vocab_sz = 1024
gmm_vocab_sz = 1024
hidden_dim = 512
# vq_dim = 128
commit_loss_weight = 0.2
epochs = 20000
log_freq = 100
train_split = 0.9
device='cuda:0'
torch.manual_seed(0)

surface_feat = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/surface_feat.npy') #[B, m, d_surface]
raw_gmm = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/raw_gmm.npy') #[B, 1, m, 16] 
surface_feat_shape = surface_feat.shape
gmm_shape = raw_gmm.shape

# pool [B*m, feat]
surface_feat = surface_feat.reshape(-1, surface_feat.shape[-1]) #[B*m, d_surface]
raw_gmm = raw_gmm.reshape(-1, raw_gmm.shape[-1]) #[B*m, 16] 



# train loop
name2feats = {
    'surface_feat': (surface_feat, surface_vocab_sz),
    'raw_gmm': (raw_gmm, gmm_vocab_sz),
}
for feat_name, (continuous_feat, vocab_sz) in name2feats.items():
    vq_dim = continuous_feat.shape[-1]
    print(f"{feat_name} VQ feat dim: {vq_dim}")
    autoencoder = VQAutoEncoder(vocab_sz, continuous_feat.shape[-1], hidden_dim, vq_dim)
    autoencoder.train()
    autoencoder.to(device)
    opt = torch.optim.Adam(autoencoder.parameters())
    running_train_loss = 0
    running_val_loss = 0
    continuous_feat = torch.tensor(continuous_feat).to(device)

    # train/val split
    B = continuous_feat.shape[0]
    randperm = torch.randperm(B)
    train_feat = continuous_feat[randperm[:int(train_split*B)]]
    val_feat = continuous_feat[randperm[int(train_split*B):]]
    print(f"test/val: {train_feat.shape[0]}/{val_feat.shape[0]}")

    for i in range(epochs): # 1 epoch = 1 step since data fits in mem
        opt.zero_grad()
        recon_feat, indices, commit_loss = autoencoder(train_feat)
        loss = nn.functional.mse_loss(train_feat, recon_feat) + commit_loss_weight*commit_loss
        loss.backward()
        opt.step()

        # val
        with torch.no_grad():
            autoencoder.eval()
            val_recon_feat, _, val_commit_loss = autoencoder(val_feat)
            val_loss = nn.functional.mse_loss(val_feat, val_recon_feat) + commit_loss_weight*val_commit_loss
            autoencoder.train()


        running_train_loss += loss.item()
        running_val_loss += val_loss.item()
        if i % log_freq == 1:
            avg_train_loss = running_train_loss / log_freq
            avg_val_loss = running_val_loss / log_freq
            print(f"batch {i}  train : {avg_train_loss}")
            print(f"           val   : {avg_val_loss}")
            running_train_loss = 0
            running_val_loss = 0
    
    # get codebook, code indices, quantized (reconstructed) feats
    autoencoder.eval()
    codebook = autoencoder.get_codebook()
    _, code_indices, _ = autoencoder(continuous_feat)
    # equivalently, get reconstructed quantized feats by indexing into codebook then applying decoder
    final_recon_feat = autoencoder.decode_from_quantized_latent(codebook[code_indices])

    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/{feat_name}_codebook.npy', codebook.cpu().numpy())
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/{feat_name}_code_indices.npy', code_indices.cpu().numpy())
    # NOTE: save final quantized code, not latent quantized code
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/quantized_{feat_name}.npy', final_recon_feat.detach().cpu().numpy())
    # save decoder
    torch.save(autoencoder.state_dict(), f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/autoencoder.pt')
