import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt

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
    print("# feats: ", surface_feats.shape[0])

    # form 2 codebooks
    # 1) surface feats
    kmeans = MiniBatchKMeans(n_clusters=surface_feat_vocab_sz, max_iter=400, tol=1e-5, random_state=1, batch_size=12000, n_init=5).fit(surface_feats)
    surface_feat_codebook = kmeans.cluster_centers_ # [vocab, feat]
    surface_feat_code_indices = kmeans.predict(surface_feats)
    surface_feat_quantized_codes = surface_feat_codebook[surface_feat_code_indices].reshape(surface_feats_shape)
    # print("surf codebook: ", surface_feat_codebook.shape)
    # print("surf indices:")
    # print(surface_feat_code_indices.reshape((surface_feats_shape[:-1])))
    # print("surf quantized: ", surface_feat_quantized_codes.shape)


    # TEMP: eval average distance of each feat to centroid
    print("avg cluster dist")
    surface_NN_dist = np.min(kmeans.transform(surface_feats), axis=1)  #[N_feats, n_clusters] -> #[N_feats,]
    print(f"surface:\t mean {np.mean(surface_NN_dist).round(4)} \t median {np.median(surface_NN_dist).round(4)}")

    kmeans = MiniBatchKMeans(n_clusters=gmm_vocab_sz, max_iter=400, tol=1e-5, random_state=1, batch_size=12000, n_init=5).fit(raw_gmms)
    gmm_codebook = kmeans.cluster_centers_ # [vocab, feat]
    gmm_code_indices = kmeans.predict(raw_gmms)
    gmm_quantized_codes = gmm_codebook[gmm_code_indices].reshape(gmms_shape)
    # print("gmm codebook: ", gmm_codebook.shape)
    # print("gmm indices:")
    # print(gmm_code_indices.reshape((gmms_shape[:-1])))
    # print("gmm quantized: ", gmm_quantized_codes.shape)
    gmm_NN_dist = np.min(kmeans.transform(raw_gmms), axis=1)  #[N_feats, n_clusters] -> #[N_feats,]
    print(f"gmm:\t mean {np.mean(gmm_NN_dist).round(4)} \t median {np.median(gmm_NN_dist).round(4)}")



    return np.mean(surface_NN_dist).round(4), np.median(surface_NN_dist).round(4), \
        np.mean(gmm_NN_dist).round(4), np.median(gmm_NN_dist).round(4), 

    # save codebook and quantized codes
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/surface_feat_codebook.npy', surface_feat_codebook)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/surface_feat_code_indices.npy', surface_feat_code_indices)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/quantized_surface_feats.npy', surface_feat_quantized_codes)

    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/gmm_codebook.npy', gmm_codebook)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/gmm_code_indices.npy', gmm_code_indices)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/quantized_raw_gmms.npy', gmm_quantized_codes)



#3) Quantize s_j and raw per-GMM parameters (covariance, eigenvalues, center, mixing weight) 
def make_codebook(continuous_feats, vocab_sz, feat_name):
    """
    Args
        - continuous_feats: [..., D]

    Returns
        - codebook: [vocab_sz, D]
        - code_indices: same shape as input excluding last dim
        - quantized_codes: [..., D]
    """
    orig_shape = continuous_feats.shape
    # pool feats
    continuous_feats = continuous_feats.reshape(-1, continuous_feats.shape[-1]) # [n_feats, D] 

    kmeans = MiniBatchKMeans(n_clusters=vocab_sz, max_iter=400, tol=1e-5, random_state=1, batch_size=12000, n_init=5).fit(continuous_feats)
    codebook = kmeans.cluster_centers_ # [vocab, feat]
    code_indices = kmeans.predict(continuous_feats)
    code_indices = code_indices.reshape((orig_shape[:-1])) # same shape as input excluding last dim
    quantized_codes = codebook[code_indices].reshape(orig_shape) # same shape as continuous input

    # save codebook and quantized codes
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/{feat_name}_codebook.npy',codebook)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/{feat_name}_indices.npy', code_indices)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/quantized_{feat_name}.npy', quantized_codes)
    print(f"{feat_name}_codebook: {codebook.shape}")

    print("avg cluster dist")
    NN_dist = np.min(kmeans.transform(continuous_feats), axis=1)  #[N_feats, n_clusters] -> #[N_feats,]
    mean_dist = np.mean(NN_dist).round(4)
    median_dist = np.median(NN_dist).round(4)
    print(f"\t mean {mean_dist} \t median {median_dist}")

    return codebook, code_indices, quantized_codes,mean_dist, median_dist


def quantize_surface_and_split_gmm(surface_feat_vocab_sz, cov_vocab_sz, eigenval_vocab_sz, center_vocab_sz, mix_weight_vocab_sz, sample_dir_name):
    surface_feats = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/surface_feats.npy') #[B, m, d_surface]
    covariances = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/pre_orthog_covariances.npy') #[B,1,m,9]
    eigenvals = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/eigenvalues.npy') # [B,1,m,3] 
    centers = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/centers.npy') # [B,1,m,3] 
    mix_weights = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/mixing_weights.npy') # [B,1,m]
    mix_weights = np.expand_dims(mix_weights, -1) # [B,1,m,1]

    # form 5 codebooks
    # # 1) surface feats
    # surface_feat_codebook, surface_feat_code_indices = \
    #     make_codebook(surface_feats, surface_feat_vocab_sz,)
    # surface_feat_quantized_codes = surface_feat_codebook[surface_feat_code_indices].reshape(surface_feats_shape) # same shape as continuous input
    # surface_feat_code_indices = surface_feat_code_indices.reshape((surface_feats_shape[:-1])) # same shape as input excluding last dim
    # print("surf codebook: ", surface_feat_codebook.shape)
    # print("surf indices:")
    # print(surface_feat_code_indices[:5])
    # # save codebook and quantized codes
    # np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/surface_feat_codebook.npy', surface_feat_codebook)
    # np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/surface_feat_code_indices.npy', surface_feat_code_indices)
    # np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/quantized_surface_feats.npy', surface_feat_quantized_codes)

    surface_mean_dist, surface_median_dist = make_codebook(surface_feats, surface_feat_vocab_sz, "surface_feats")[-2:]
    eigenvec_mean_dist, eigenvec_median_dist = make_codebook(covariances, cov_vocab_sz, "pre_orthog_covariances")[-2:]
    eigenval_mean_dist, eigenval_median_dist = make_codebook(eigenvals, eigenval_vocab_sz, "eigenvalues")[-2:]
    center_mean_dist, center_median_dist = make_codebook(centers, center_vocab_sz, "centers")[-2:]
    mixing_weight_mean_dist, mixing_weight_median_dist = make_codebook(mix_weights, mix_weight_vocab_sz, "mixing_weights")[-2:]

    return [surface_mean_dist, surface_median_dist],\
        [eigenvec_mean_dist, eigenvec_median_dist],\
        [eigenval_mean_dist, eigenval_median_dist],\
        [center_mean_dist, center_median_dist],\
        [mixing_weight_mean_dist, mixing_weight_median_dist]

sample_dir_name = 'split_gmm_surface_feat_quantize_optimal'
# vocab_sz = 1024
# quantize_surface_and_raw_gmm(
#     surface_feat_vocab_sz=vocab_sz,
#     gmm_vocab_sz=vocab_sz,
#     sample_dir_name=sample_dir_name
# )
quantize_surface_and_split_gmm(
    surface_feat_vocab_sz=1500, 
    cov_vocab_sz=1500,
    eigenval_vocab_sz=1000, 
    center_vocab_sz=1300, 
    mix_weight_vocab_sz=500, 
    sample_dir_name=sample_dir_name)

# TEMP: measure avg cluster quality vs. vocab sz

# mean_surface_dists = []
# median_surface_dists = []
# mean_eigenvec_dists = []
# median_eigenvec_dists = []
# mean_eigenval_dists = []
# median_eigenval_dists = []
# mean_center_dists = []
# median_center_dists = []
# mean_mixing_weight_dists = []
# median_mixing_weight_dists = []

# vocab_sizes_small = 32*np.arange(10)[1:]
# vocab_sizes_mid = 64*np.arange(6)[1:] + vocab_sizes_small[-1]
# vocab_sizes_large = 256*np.arange(8)[1:] + vocab_sizes_mid[-1]

# vocab_sizes = np.concatenate([vocab_sizes_small, vocab_sizes_mid, vocab_sizes_large])
# print("vocabs: ",vocab_sizes)

# for vocab_sz in vocab_sizes:
#     print("vocab sz: ", vocab_sz)
#     [surface_mean_dist, surface_median_dist],\
#         [eigenvec_mean_dist, eigenvec_median_dist],\
#         [eigenval_mean_dist, eigenval_median_dist],\
#         [center_mean_dist, center_median_dist],\
#         [mixing_weight_mean_dist, mixing_weight_median_dist] = quantize_surface_and_split_gmm(vocab_sz, vocab_sz, vocab_sz, vocab_sz, vocab_sz, sample_dir_name)
#     mean_surface_dists.append(surface_mean_dist)
#     median_surface_dists.append(surface_median_dist)
#     mean_eigenvec_dists.append(eigenvec_mean_dist)
#     median_eigenvec_dists.append(eigenvec_median_dist)
#     mean_eigenval_dists.append(eigenval_mean_dist)
#     median_eigenval_dists.append(eigenval_median_dist)
#     mean_center_dists.append(center_mean_dist)
#     median_center_dists.append(center_median_dist)
#     mean_mixing_weight_dists.append(mixing_weight_mean_dist)
#     median_mixing_weight_dists.append(mixing_weight_median_dist)

# # plot cluster score vs vocab
# figure, axis = plt.subplots(nrows=3, ncols=2,figsize=(15, 15))
# axis[0][0].plot(vocab_sizes, mean_surface_dists, marker='o', label="mean")
# axis[0][0].plot(vocab_sizes, median_surface_dists, marker='o', label="median")
# axis[0][0].set_xlabel("Codebook size")
# axis[0][0].set_ylabel("NN distance")
# axis[0][0].set_title("Intrinsic Vector Nearest Neighbor Distance (lower is better)")
# axis[0][0].legend()

# axis[0][1].plot(vocab_sizes, mean_eigenvec_dists, marker='o', label="mean")
# axis[0][1].plot(vocab_sizes, median_eigenvec_dists, marker='o', label="median")
# axis[0][1].set_xlabel("Codebook size")
# axis[0][1].set_ylabel("NN distance")
# axis[0][1].set_title("Covariance Basis Nearest Neighbor Distance")
# axis[0][1].legend()

# axis[1][0].plot(vocab_sizes, mean_eigenval_dists, marker='o', label="mean")
# axis[1][0].plot(vocab_sizes, median_eigenval_dists, marker='o', label="median")
# axis[1][0].set_xlabel("Codebook size")
# axis[1][0].set_ylabel("NN distance")
# axis[1][0].set_title("Covariance Eigenvalue Nearest Neighbor Distance")
# axis[1][0].legend()

# axis[1][1].plot(vocab_sizes, mean_center_dists, marker='o', label="mean")
# axis[1][1].plot(vocab_sizes, median_center_dists, marker='o', label="median")
# axis[1][1].set_xlabel("Codebook size")
# axis[1][1].set_ylabel("NN distance")
# axis[1][1].set_title("Gaussian Center Nearest Neighbor Distance")
# axis[1][1].legend()

# axis[2][0].plot(vocab_sizes, mean_mixing_weight_dists, marker='o', label="mean")
# axis[2][0].plot(vocab_sizes, median_mixing_weight_dists, marker='o', label="median")
# axis[2][0].set_xlabel("Codebook size")
# axis[2][0].set_ylabel("NN distance")
# axis[2][0].set_title("Mixing Weight Nearest Neighbor Distance")
# axis[2][0].legend()

# plt.savefig("experiments/surface_split_gmm_cluster_distances.jpg")
# plt.show()
