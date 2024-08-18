import numpy as np   


sample_dir_name='rand_unconditional'


train_surface_feats = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_train/surface_feats.npy') #[B, m, d_surface]
train_covariances = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_train/pre_orthog_covariances.npy') #[B,1,m,9]
train_eigenvals = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_train/eigenvalues.npy') # [B,1,m,3] 
train_centers = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_train/centers.npy') # [B,1,m,3] 
train_mix_weights = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_train/mixing_weights.npy') # [B,1,m]
# train_mix_weights = np.expand_dims(train_mix_weights, -1) # [B,1,m,1]

rand_surface_feats = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_unconditional/surface_feats.npy') #[B, m, d_surface]
rand_covariances = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_unconditional/pre_orthog_covariances.npy') #[B,1,m,9]
rand_eigenvals = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_unconditional/eigenvalues.npy') # [B,1,m,3] 
rand_centers = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_unconditional/centers.npy') # [B,1,m,3] 
rand_mix_weights = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_unconditional/mixing_weights.npy') # [B,1,m]
# rand_mix_weights = np.expand_dims(rand_mix_weights, -1) # [B,1,m,1]


surface_feats = np.concatenate([train_surface_feats, rand_surface_feats], axis=0)
covariances = np.concatenate([train_covariances, rand_covariances], axis=0)
eigenvals = np.concatenate([train_eigenvals, rand_eigenvals], axis=0)
centers = np.concatenate([train_centers, rand_centers], axis=0)
mix_weights = np.concatenate([train_mix_weights, rand_mix_weights], axis=0)
print(surface_feats.shape)
print(covariances.shape)
print(eigenvals.shape)
print(centers.shape)
print(mix_weights.shape)

np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_combined/surface_feats.npy', surface_feats)
np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_combined/pre_orthog_covariances.npy', covariances)
np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_combined/eigenvalues.npy', eigenvals)
np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_combined/centers.npy', centers)
np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes_combined/mixing_weights.npy', mix_weights)