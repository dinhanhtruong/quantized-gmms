import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

#1) Quantize z_b vectors (which are projected to s_j and GMMs)
def quantize_zb(vocab_sz, sample_dir_name):
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
def quantize_surface_and_raw_gmm(surface_feat_vocab_sz, gmm_vocab_sz, sample_dir_name):
    surface_feats = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/surface_feats.npy') #[B, m, d_surface]
    raw_gmms = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/raw_gmms.npy') #[B, 1, m, 16] 
    
    surface_feats_shape = surface_feats.shape
    gmms_shape = raw_gmms.shape

    # pool [B*m, feat]
    surface_feats = surface_feats.reshape(-1, surface_feats.shape[-1]) #[B*m, d_surface]
    raw_gmms = raw_gmms.reshape(-1, raw_gmms.shape[-1]) #[B*m, 16] 

    # form 2 codebooks
    # 1) surface feats
    kmeans = MiniBatchKMeans(n_clusters=surface_feat_vocab_sz, max_iter=400, tol=1e-5, random_state=1, batch_size=8192, n_init=10).fit(surface_feats)
    surface_feat_codebook = kmeans.cluster_centers_ # [vocab, feat]
    surface_feat_code_indices = kmeans.predict(surface_feats)
    surface_feat_quantized_codes = surface_feat_codebook[surface_feat_code_indices].reshape(surface_feats_shape)
    print("surf codebook: ", surface_feat_codebook.shape)
    print("surf indices:")
    print(surface_feat_code_indices.reshape((surface_feats_shape[:-1])))
    print("surf quantized: ", surface_feat_quantized_codes.shape)

    kmeans = MiniBatchKMeans(n_clusters=gmm_vocab_sz, max_iter=400, tol=1e-5, random_state=1, batch_size=8192, n_init=10).fit(raw_gmms)
    gmm_codebook = kmeans.cluster_centers_ # [vocab, feat]
    gmm_code_indices = kmeans.predict(raw_gmms)
    gmm_quantized_codes = gmm_codebook[gmm_code_indices].reshape(gmms_shape)
    print("gmm codebook: ", gmm_codebook.shape)
    print("gmm indices:")
    print(gmm_code_indices.reshape((gmms_shape[:-1])))
    print("gmm quantized: ", gmm_quantized_codes.shape)

    # save codebook and quantized codes
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/surface_feat_codebook.npy', surface_feat_codebook)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/surface_feat_code_indices.npy', surface_feat_code_indices)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/quantized_surface_feats.npy', surface_feat_quantized_codes)

    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/gmm_codebook.npy', gmm_codebook)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/gmm_code_indices.npy', gmm_code_indices)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/quantized_raw_gmms.npy', gmm_quantized_codes)



#3) Quantize s_j and raw per-GMM parameters (covariance, eigenvalues, center, mixing weight) 


vocab_sz = 512 #1024
sample_dir_name = 'raw_gmm_surface_feat_quantize'
quantize_surface_and_raw_gmm(surface_feat_vocab_sz=vocab_sz, gmm_vocab_sz=vocab_sz, sample_dir_name=sample_dir_name)