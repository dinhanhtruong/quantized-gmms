import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

#1) Quantize z_b vectors (which are projected to s_j and GMMs)
vocab_sz = 1024
sample_dir_name = 'zb_quantize'

z_b = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/all_zb_base.npy') # [B, m, dim_h]
# pool feats [B*m, feat]
continuous_codes = z_b.reshape(-1, z_b.shape[-1])
kmeans = MiniBatchKMeans(n_clusters=vocab_sz, max_iter=400, tol=1e-5, random_state=1, batch_size=8192, n_init=10).fit(continuous_codes)
codebook = kmeans.cluster_centers_ # [vocab, feat]

code_indices = kmeans.predict(continuous_codes)
quantized_codes = codebook[code_indices].reshape(z_b.shape)

print("codebook: ", codebook.shape)
# print(codebook[:, :5])
# print("input: ")
# print(z_b[..., :5])
print("indices:")
print(code_indices.reshape((z_b.shape[:-1])))
print("quantized: ", quantized_codes.shape)

# save codebook and quantized codes
np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/codebook.npy', codebook)
np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/codebook_indices.npy', code_indices)
np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/quantized_zb.npy', quantized_codes)


#2) Quantize s_j and raw stacked GMM vec
#3) Quantize s_j and raw per-GMM parameters (covariance, eigenvalues, center, mixing weight) 
