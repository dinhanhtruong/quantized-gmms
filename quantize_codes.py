import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
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

    # form 2 codebooks
    # 1) surface feats
    kmeans = MiniBatchKMeans(n_clusters=surface_feat_vocab_sz, max_iter=400, tol=1e-5, random_state=1, batch_size=12000, n_init=10).fit(surface_feats)
    surface_feat_codebook = kmeans.cluster_centers_ # [vocab, feat]
    surface_feat_code_indices = kmeans.predict(surface_feats)
    surface_feat_quantized_codes = surface_feat_codebook[surface_feat_code_indices].reshape(surface_feats_shape)
    # print("surf codebook: ", surface_feat_codebook.shape)
    # print("surf indices:")
    # print(surface_feat_code_indices.reshape((surface_feats_shape[:-1])))
    # print("surf quantized: ", surface_feat_quantized_codes.shape)


    # TEMP: measure cluster quality
    surface_clustering_CH = metrics.calinski_harabasz_score(surface_feats, kmeans.labels_)
    surface_clustering_silhouette = metrics.silhouette_score(surface_feats, kmeans.labels_, metric='euclidean')
    print("surface CH: ", surface_clustering_CH)
    print("surface Silhouette: ", surface_clustering_silhouette)

    kmeans = MiniBatchKMeans(n_clusters=gmm_vocab_sz, max_iter=400, tol=1e-5, random_state=1, batch_size=12000, n_init=10).fit(raw_gmms)
    gmm_codebook = kmeans.cluster_centers_ # [vocab, feat]
    gmm_code_indices = kmeans.predict(raw_gmms)
    gmm_quantized_codes = gmm_codebook[gmm_code_indices].reshape(gmms_shape)
    # print("gmm codebook: ", gmm_codebook.shape)
    # print("gmm indices:")
    # print(gmm_code_indices.reshape((gmms_shape[:-1])))
    # print("gmm quantized: ", gmm_quantized_codes.shape)

    gmm_clustering_CH = metrics.calinski_harabasz_score(raw_gmms, kmeans.labels_)
    gmm_clustering_silhouette = metrics.silhouette_score(raw_gmms, kmeans.labels_, metric='euclidean')
    print("GMM CH: ", gmm_clustering_CH)
    print("GMM Silhouette: ", gmm_clustering_silhouette)
    return surface_clustering_CH, surface_clustering_silhouette, gmm_clustering_CH, gmm_clustering_silhouette


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
        - code_indices: [n_feats,]
        - quantized_codes: [..., D]
    """
    orig_shape = continuous_feats.shape
    # pool feats
    continuous_feats = continuous_feats.reshape(-1, continuous_feats.shape[-1]) # [n_feats, D] 

    kmeans = MiniBatchKMeans(n_clusters=vocab_sz, max_iter=400, tol=1e-5, random_state=1, batch_size=12000, n_init=10).fit(continuous_feats)
    codebook = kmeans.cluster_centers_ # [vocab, feat]
    code_indices = kmeans.predict(continuous_feats)
    code_indices = code_indices.reshape((orig_shape[:-1])) # same shape as input excluding last dim
    quantized_codes = codebook[code_indices].reshape(orig_shape) # same shape as continuous input

    # save codebook and quantized codes
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/{feat_name}_codebook.npy',codebook)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/{feat_name}_indices.npy', code_indices)
    np.save(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/quantized_{feat_name}.npy', quantized_codes)
    print(f"{feat_name}_codebook: {codebook.shape}")
    return codebook, code_indices, quantized_codes


def quantize_surface_and_split_gmm(surface_feat_vocab_sz, cov_vocab_sz, eigenval_vocab_sz, center_vocab_sz, mix_weight_vocab_sz, sample_dir_name):
    surface_feats = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/surface_feats.npy') #[B, m, d_surface]
    covariances = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/pre_orthog_covariances.npy') #[B,1,m,9]
    eigenvals = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/eigenvalues.npy') # [B,1,m,3] 
    centers = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/centers.npy') # [B,1,m,3] 
    mix_weights = np.load(f'assets/checkpoints/spaghetti_airplanes/{sample_dir_name}/codes/mixing_weights.npy') # [B,1,m]


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

    make_codebook(surface_feats, surface_feat_vocab_sz, "surface_feats")
    make_codebook(covariances, cov_vocab_sz, "pre_orthog_covariances")
    make_codebook(eigenvals, eigenval_vocab_sz, "eigenvalues")
    make_codebook(centers, center_vocab_sz, "centers")
    make_codebook(mix_weights, mix_weight_vocab_sz, "mixing_weights")

    

# vocab_sz = 128
sample_dir_name = 'raw_gmm_surface_feat_quantize_512'

intrinsic_silhouette = []
extrinsic_silhouette = []
intrinsic_CH = []
extrinsic_CH = []
vocab_sizes = [64, 128, 256, 384, 512, 640, 768, 1024,2048,3072]
for vocab_sz in vocab_sizes:
    print("vocab sz: ", vocab_sz)
    surface_clustering_CH, surface_clustering_silhouette, gmm_clustering_CH, gmm_clustering_silhouette = quantize_surface_and_raw_gmm(vocab_sz, vocab_sz, sample_dir_name)
    intrinsic_CH.append(surface_clustering_CH)
    extrinsic_CH.append(gmm_clustering_CH)
    intrinsic_silhouette.append(surface_clustering_silhouette)
    extrinsic_silhouette.append(gmm_clustering_silhouette)

# plot cluster score vs vocab
figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(15, 5))
axis[0].plot(vocab_sizes, intrinsic_CH, marker='o')
axis[0].set_xlabel("Codebook size")
axis[0].set_ylabel("CH Index")
axis[0].set_title("Intrinsic vector CH Index (higher is better)")
axis[1].plot(vocab_sizes, intrinsic_silhouette, marker='o')
axis[1].set_xlabel("Codebook size")
axis[1].set_ylabel("Silhouette Coeff")
axis[1].set_title("Intrinsic vector Silhouette Coeff. (higher is better)")
plt.savefig("experiments/surface_cluster_scores.jpg")
plt.close()

figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(15, 5))
axis[0].plot(vocab_sizes, extrinsic_CH, marker='o')
axis[0].set_xlabel("Codebook size")
axis[0].set_ylabel("CH Index")
axis[0].set_title("Extrinsic vector CH Index (higher is better)")
axis[1].plot(vocab_sizes, extrinsic_silhouette, marker='o')
axis[1].set_xlabel("Codebook size")
axis[1].set_ylabel("Silhouette Coeff")
axis[1].set_title("Extrinsic vector Silhouette Coeff. (higher is better)")
plt.savefig("experiments/raw_gmm_cluster_scores.jpg")
plt.show()

# quantize_surface_and_split_gmm(
#     surface_feat_vocab_sz=vocab_sz, 
#     cov_vocab_sz=vocab_sz,
#     eigenval_vocab_sz=vocab_sz, 
#     center_vocab_sz=vocab_sz, 
#     mix_weight_vocab_sz=vocab_sz, 

#     sample_dir_name=sample_dir_name)