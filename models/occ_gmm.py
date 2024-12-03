import os
import random
from options import Options, recon_sample_offset
import options
from models import models_utils, transformer
import constants
from custom_types import *
from torch import distributions
import math
from utils import files_utils

def dot(x, y, dim=3):
    return torch.sum(x * y, dim=dim)


def remove_projection(v_1, v_2):
    proj = (dot(v_1, v_2) / dot(v_2, v_2))
    return v_1 - proj[:, :, :, None] * v_2


def get_p_direct(splitted: TS) -> T:
    '''
    p sure this forms an orthonormal basis (via Gram-schmidt?) in a 3x3 from the raw linear output
        i.e. factorized covariance matrix (unitary)
    '''
    raw_base = []
    for i in range(constants.DIM): # 3 dims
        u = splitted[i]
        for j in range(i):
            u = remove_projection(u, raw_base[j])
        raw_base.append(u)
    p = torch.stack(raw_base, dim=3)
    p = p / torch.norm(p, p=2, dim=4)[:, :, :, :, None]  # + self.noise[None, None, :, :]
    return p

def load_feats_from_indices(filepath, feat_name, tf_sample_dirname=''):
    # prefix for TF outputs
    prefix = 'sampled_' if tf_sample_dirname else ''
    # get indices
    indices = np.load(f'{filepath}/{tf_sample_dirname}/{prefix}{feat_name}_indices.npy')[options.recon_sample_offset:options.recon_sample_offset+22000] #[B, 1, m]
    # get codebook
    codebook = np.load(f'{filepath}/{feat_name}_codebook.npy') #[vocab_sz, feat_dim]
    # index into codebook
    print("indexed feats: ", codebook[indices].shape)
    return torch.tensor(codebook[indices]).cuda()

def split_gm(splitted: TS, output_dir='', tf_sample_dirname='', saved_codes_dirname='codes_combined') -> TS:
    '''
    args:
        - splitted: list of tensors with shapes 5x[B,1,m,3], [B,1,m,1]
                where first three tensors form cov matrix, then eigenvalues, centroids, and mixing weights
    
    returns:
        - mu=centroids  [B,1,m,3]
        - p=factorized cov matrices [B, 1, m, 3,3]
        - phi=mixing weights [B,1,m]
        - eigenvals  [B,1,m,3]
    '''
    # quantize pre-orthogonalized cov (concat first 3 tensors in splitted to get [B,1,m,9])
    pre_orthog_cov = torch.cat(splitted[:3], dim=-1) # [B,1,m,9]

    eigen = splitted[-3] ** 2 + constants.EPSILON # eigenvalues of diag matrix (lambda in paper). [B,1,m,3]
    mu = splitted[-2]  # 3D centroid [B,1,m,3]
    phi = splitted[-1].squeeze(3)  # mixing constants/scale factor. [B,1,m]
    # all_centroids = mu.view(-1, 3) # [total_parts, 3]
    # all_eigenvals = eigen.view(-1, 3) # [total_parts, 3]

    # AT: 3rd quantization scheme, pt 2 (per-GMM param + s_j)
    assert output_dir
    save_path = f'assets/checkpoints/spaghetti_airplanes/{output_dir}/codes/'
    if not os.path.exists(f'{save_path}/pre_orthog_covariances.npy'):
        print("saving new per-param codes")
        np.save(f'{save_path}/pre_orthog_covariances.npy', pre_orthog_cov.detach().cpu().numpy()) #[B,1,m,9]
        np.save(f'{save_path}/eigenvalues.npy', eigen.detach().cpu().numpy()) # squared eigenvals, 3 per part [B,1,m,3] 
        np.save(f'{save_path}/centers.npy', mu.detach().cpu().numpy()) # [B,1,m,3]
        np.save(f'{save_path}/mixing_weights.npy', phi.detach().cpu().numpy()) # [B,1,m]
    else:
        assert not options.use_quantized
        if options.use_quantized:
            # load QUANTIZED codes from disk
            print('loading existing per-param quantized codes')
            pre_orthog_cov = load_feats_from_indices(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/codes/', 'pre_orthog_covariances', tf_sample_dirname)
            eigen = load_feats_from_indices(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/codes/', 'eigenvalues', tf_sample_dirname)
            mu = load_feats_from_indices(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/codes/', 'centers', tf_sample_dirname)
            phi = load_feats_from_indices(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/codes/', 'mixing_weights', tf_sample_dirname)
            phi = phi.view(phi.shape[:3]) # [B,1,16,1] -> [B,1,16]
            # # split quantized pre-orthog cov into column vectors
            # splitted = list(splitted)
            # splitted[0] = pre_orthog_cov[..., :3]
            # splitted[1] = pre_orthog_cov[..., 3:6]
            # splitted[2] = pre_orthog_cov[..., 6:] # remaining elements of splitted aren't used in get_p_direct
            # splitted = tuple(splitted)
        else:
            # load continuous codes 
            print('loading existing continuous codes')
            pre_orthog_cov = torch.tensor(np.load(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/{saved_codes_dirname}/pre_orthog_covariances.npy')).cuda()
            eigen = torch.tensor(np.load(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/{saved_codes_dirname}/eigenvalues.npy')).cuda()
            mu = torch.tensor(np.load(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/{saved_codes_dirname}/centers.npy')).cuda()
            phi = torch.tensor(np.load(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/{saved_codes_dirname}/mixing_weights.npy')).cuda()
            # phi = phi.view(phi.shape[:3]) # [B,1,16,1] -> [B,1,16]
        # split saved pre-orthog cov into column vectors
        splitted = list(splitted)
        splitted[0] = pre_orthog_cov[..., :3]
        splitted[1] = pre_orthog_cov[..., 3:6]
        splitted[2] = pre_orthog_cov[..., 6:] # remaining elements of splitted aren't used in get_p_direct
        splitted = tuple(splitted)

    p = get_p_direct(splitted) # 3x3 factorized covariance matrix [B, 1, m, 3,3]
    print('p:', p.shape)
    print('eig:', eigen.shape)
    print('mu:', mu.shape)
    print('phi:', phi.shape)

    return mu, p, phi, eigen

class DecompositionNetwork(nn.Module):

    def forward_bottom(self, x):
        return self.l1(x).view(-1, self.bottom_width, self.embed_dim)

    def forward_upper(self, x):
        return self.to_zb(x)

    def forward(self, x):
        x = self.forward_bottom(x) # split into m vecs
        x = self.forward_upper(x) # shared MLP across m parts
        return x # z_b

    def __init__(self, opt: Options, act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm):
        super(DecompositionNetwork, self).__init__()
        self.bottom_width = opt.num_gaussians
        self.embed_dim = opt.dim_h
        self.l1 = nn.Linear(opt.dim_z, self.bottom_width * opt.dim_h)
        if opt.decomposition_network == 'mlp':

            self.to_zb = models_utils.MLP((opt.dim_h, *([2 * opt.dim_h] * opt.decomposition_num_layers), opt.dim_h))
        else:
            self.to_zb = transformer.Transformer(opt.dim_h, opt.num_heads, opt.num_layers, act=act,
                                                       norm_layer=norm_layer)


class OccupancyMlP(nn.Module):
    ## base on DeepSDF https://github.com/facebookresearch/DeepSDF
    def forward(self, x, z):
        x_ = x = torch.cat((x, z), dim=-1)
        for i, layer in enumerate(self.layers):
            if layer == self.latent_in:
                x = torch.cat([x, x_], 2)
            x = layer(x)
            if i < len(self.layers) - 2:
                x = self.relu(x)
                # x = self.dropout(self.relu(x))
            # files_utils.save_pickle(x.detach().cpu(), f"/home/amirh/projects/spaghetti_private/assets/debug/out_{i}")
        return x

    def __init__(self, opt: Options):
        super(OccupancyMlP, self).__init__()
        dim_in = 2 * (opt.pos_dim + constants.DIM)
        dims = [dim_in] + opt.head_occ_size * [dim_in] + [1]
        self.latent_in = opt.head_occ_size // 2 + opt.head_occ_size % 2
        dims[self.latent_in] += dims[0]
        self.dropout = nn.Dropout(.2)
        self.relu = nn.ReLU(True)
        layers = []
        for i in range(0, len(dims) - 1):
            layers.append(nn.utils.weight_norm(nn.Linear(dims[i], dims[i + 1])))
        self.layers = nn.ModuleList(layers)


class OccupancyNetwork(nn.Module):

    def get_pos(self, coords: T):
        pos = self.pos_encoder(coords)
        pos = torch.cat((coords, pos), dim=2)
        return pos

    def forward_attention(self, coords: T, zh: T, mask: Optional[T] = None, alpha: TN = None) -> TS:
        pos = self.get_pos(coords)
        _, attn = self.occ_transformer.forward_with_attention(pos, zh, mask, alpha)
        return attn

    def forward(self, coords: T, zh: T,  mask: TN = None, alpha: TN = None) -> T:
        pos = self.get_pos(coords)
        x = self.occ_transformer(pos, zh, mask, alpha)
        out = self.occ_mlp(pos, x)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def __init__(self, opt: Options):
        super(OccupancyNetwork, self).__init__()
        self.pos_encoder = models_utils.SineLayer(constants.DIM, opt.pos_dim, is_first=True)

        if hasattr(opt, 'head_occ_type') and opt.head_occ_type == 'skip':
            self.occ_mlp = OccupancyMlP(opt)
        else:
            self.occ_mlp = models_utils.MLP([(opt.pos_dim + constants.DIM)] +
                                            [opt.dim_h] * opt.head_occ_size + [1])
        self.occ_transformer = transformer.Transformer(opt.pos_dim + constants.DIM,
                                                       opt.num_heads_head, opt.num_layers_head,
                                                       dim_ref=opt.dim_h)

class DecompositionControl(models_utils.Model):

    def forward_bottom(self, x):
        z_bottom = self.decomposition.forward_bottom(x)
        return z_bottom

    def forward_upper(self, x):
        x = self.decomposition.forward_upper(x)
        return x

    def forward_split(self, x: T, output_dir='', tf_sample_dirname='', saved_codes_dirname='codes_combined') -> Tuple[T, TS]:
        '''
        AT

        high dim per-part vector z_b to {surface detail vec, 16D GMM params}
        x: [B, m, dim_h]

        returns
            -surface detail vec s_j=zh_base: [B, m, d_surface]
            -gmms: centroids, factorized_cov, mixing_weights, eigenvals
        '''
        b = x.shape[0]
        # NOTE: with reflection symmetry on, only the first half of m gaussians matter, since they get reflected and concat'ed. 
            # the remaining half are discarded (although their surface vecs remain)
        raw_gmm = self.to_gmm(x).unsqueeze(1)  #linear. [B, 1, m, 16] 
        raw_gmm_shape= raw_gmm.shape  #linear. [B, 1, m, 16] 
        zh = self.to_s(x) #linear
        zh = zh.view(b, -1, zh.shape[-1]) #[B, m, d_surface]
        zh_shape = zh.shape #[B, m, d_surface]

        # if options.use_salad_data:
        #     print("using SALAD-generated priming shapes")
        #     tuple_indices = torch.load(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/codes/salad_val_indices.pt')
        #     tuple_index = tuple_indices[0]
        #     raw_gmm = torch.load(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/codes/salad_gmm.pt') # [B, k=5, m=16, 16]
        #     zh = torch.load(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/codes/salad_intrinsic.pt') # [B, k=5, m=16, dim=512]

        #     raw_gmm = raw_gmm[tuple_index].unsqueeze(1)  # [k=5, 1, m=16, 16]
        #     zh = zh[tuple_index] # [k=5, m=16, dim=512]



        # AT: 3rd quantization scheme, part 1 (per-GMM param + s_j)
        assert output_dir
        save_path = f'assets/checkpoints/spaghetti_airplanes/{output_dir}/{saved_codes_dirname}/'

        if not os.path.exists(f'{save_path}/surface_feats.npy'):
            print("saving new codes")
            os.makedirs(save_path)
            np.save(f'{save_path}/surface_feats.npy', zh.detach().cpu().numpy()) #[B, m, d_surface]

            # save stacked gmm
            # print("saving ", raw_gmm.squeeze().shape)
            # np.save(f'{save_path}/stacked_gaussians.npy', raw_gmm.squeeze().detach().cpu().numpy()) #[B, m, 16]  
        else:
            assert not options.use_quantized
            # if options.use_quantized:
            #     # load from disk
            #     # print('loading existing quantized codes')
            #     zh = load_feats_from_indices(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/codes', 'surface_feats', tf_sample_dirname=tf_sample_dirname)
            #     if zh.dim() == 4: # incompatibility b/t TF indices and quantized indices
            #         zh = zh[:, 0, :, :] #[B, 1, m, d_surface] -> #[B, m, d_surface].  
                    
            #     # zh = np.load(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/codes/quantized_surface_feats.npy')
            #     # zh = torch.tensor(zh).cuda()
            # else:
            zh = torch.tensor(np.load(f'assets/checkpoints/spaghetti_airplanes/{output_dir}/{saved_codes_dirname}/surface_feats.npy')).cuda()
            if zh.dim() == 4: # incompatibility b/t TF indices and quantized indices
                zh = zh[:, 0, :, :] #[B, 1, m, d_surface] -> #[B, m, d_surface].  

        gmms = split_gm(torch.split(raw_gmm, self.split_shape, dim=3), output_dir=output_dir, tf_sample_dirname=tf_sample_dirname)
        return zh, gmms

    @staticmethod
    def apply_gmm_affine(gmms: TS, affine: T):
        '''
        transform centroid pos and factorized cov matrix
        '''
        mu, p, phi, eigen = gmms
        if affine.dim() == 2:
            affine = affine.unsqueeze(0).expand(mu.shape[0], *affine.shape)
        mu_r = torch.einsum('bad, bpnd->bpna', affine, mu)
        p_r = torch.einsum('bad, bpncd->bpnca', affine, p)
        return mu_r, p_r, phi, eigen

    @staticmethod
    def concat_gmm(gmm_a: TS, gmm_b: TS):
        '''
        Concats half the gaussians from each of the arguments
        '''
        out = []
        num_gaussians = gmm_a[0].shape[2] // 2
        # gaussian_perm = torch.randperm(16)[:8]
        for element_a, element_b in zip(gmm_a, gmm_b):
            # for each parameter type, get the ones for the first num_gaussians from each arg
            out.append(torch.cat((element_a[:, :, :num_gaussians], element_b[:, :, :num_gaussians]), dim=2))
            # out.append(torch.cat((element_a[:, :, gaussian_perm], element_b[:, :, gaussian_perm]), dim=2))
        return out

    def forward_mid(self, zs, output_dir='', tf_sample_dirname='') -> Tuple[T, TS]:
        '''
        Args
            - Z_b: [B, m, dim_h]
        '''
        zh, gmms = self.forward_split(zs, output_dir, tf_sample_dirname)  # split z_b into surface vec + gaussian params
        if self.reflect is not None:
            print("REFLECTING")
            gmms_r = self.apply_gmm_affine(gmms, self.reflect)
            gmms = self.concat_gmm(gmms, gmms_r)
        return zh, gmms

    def forward_low(self, z_init):
        zs = self.decomposition(z_init)
        return zs

    def forward(self, z_init, output_dir='', tf_sample_dirname='') -> Tuple[T, TS]:
        zs = self.forward_low(z_init) # split z_a into m part vectors (Z_b): [B, m, dim_h]
        # AT: 1st quantization scheme 
        # AT: save continuous z_b for quantization before splitting into surface/GMM vecs
        save_path = f'assets/checkpoints/spaghetti_airplanes/{output_dir}/codes/'
        # assert output_dir
        # if not os.path.exists(f'{save_path}/all_zb_base.npy'):
        #     os.makedirs(save_path)
        #     print("saving new codes")
        #     np.save(f'{save_path}/all_zb_base.npy', zs.detach().cpu().numpy())
    
        # temp: make combined zb for train + unconditional (run script once with --source training to save zb_train then run this with --source random)
        # print("zb unconditional: ", zs)
        # np.save(f'{save_path}/zb_unconditional.npy', zs.detach().cpu().numpy()) # save as zb_train.npy with --source training
        # zb_train = np.load(f'{save_path}/zb_train.npy')
        # zb_unconditional = np.load(f'{save_path}/zb_unconditional.npy')
        # zb_combined = np.concatenate([zb_train, zb_unconditional], axis=0)
        # np.save(f'{save_path}/zb_combined.npy', zb_combined)

        # old_zb_combined = torch.load(f'assets/checkpoints/spaghetti_airplanes/rand_unconditional/codes_combined/zb_combined.pt')
        # print("old zb combined :", old_zb_combined)
        # print("new zb combined :", zb_combined.shape)
        # print(zb_combined)
        # exit()

        # AT 12/2: for outlier detection, save constructed zb of sampled shapes




        # test position equivariance w/ symmetry
        # perm = torch.randperm(8)
        # zs[:, :8] = zs[:, perm]
        # zs[:, 8:] = zs[:, 8+perm]
        zh, gmms = self.forward_mid(zs, output_dir, tf_sample_dirname) # apply separate linear layers to get s_j and g_j
        return zh, gmms

    @staticmethod
    def get_reflection(reflect_axes: Tuple[bool, ...]):
        '''
        AT: returns 3x3 diag matrix where M_ii = -1 if reflect is true on that axis
        '''
        reflect = torch.eye(constants.DIM)
        for i in range(constants.DIM):
            if reflect_axes[i]:
                reflect[i, i] = -1
        return reflect

    def __init__(self, opt: Options):
        super(DecompositionControl, self).__init__()
        if sum(opt.symmetric) > 0:
            reflect = self.get_reflection(opt.symmetric)
            self.register_buffer("reflect", reflect)
        else:
            self.reflect = None
        self.split_shape = tuple((constants.DIM + 2) * [constants.DIM] + [1]) # [3,3,3,3,3,1] sums to 16
        self.decomposition = DecompositionNetwork(opt)
        self.to_gmm = nn.Linear(opt.dim_h, sum(self.split_shape))
        self.to_s = nn.Linear(opt.dim_h, opt.dim_h)


class Spaghetti(models_utils.Model):

    def get_z(self, item: T):
        return self.z(item)

    @staticmethod
    def interpolate_(z, num_between: Optional[int] = None):
        if num_between is None:
            num_between = z.shape[0]
        alphas = torch.linspace(0, 1, num_between, device=z.device)
        while alphas.dim() != z.dim():
            alphas.unsqueeze_(-1)
        z_between = alphas * z[1:2] + (- alphas + 1) * z[:1]
        return z_between

    def interpolate_higher(self, z: T, num_between: Optional[int] = None):
        z_between = self.interpolate_(z, num_between)
        zh, gmms = self.decomposition_control.forward_split(self.decomposition_control.forward_upper(z_between))
        return zh, gmms

    def interpolate(self, item_a: int, item_b: int, num_between: int):
        items = torch.tensor((item_a, item_b), dtype=torch.int64, device=self.device)
        z = self.get_z(items)
        z_between = self.interpolate_(z, num_between)
        zh, gmms = self.decomposition_control(z_between)
        return zh, gmms

    def get_disentanglement(self, items: T):
        z_a = self.get_z(items)
        z_b = self.decomposition_control.forward_bottom(z_a)
        zh, gmms = self.decomposition_control.forward_split(self.decomposition_control.forward_upper(z_b))
        return z_a, z_b, zh, gmms

    def get_embeddings(self, item: T, output_dir='', tf_sample_dirname=''):
        '''
        use train embeddings
        '''
        z = self.get_z(item)
        zh, gmms = self.decomposition_control(z, output_dir, tf_sample_dirname)
        return zh, z, gmms

    def merge_zh_step_a(self, zh, gmms):
        b, gp, g, _ = gmms[0].shape
        mu, p, phi, eigen = [item.view(b, gp * g, *item.shape[3:]) for item in gmms]
        p = p.reshape(*p.shape[:2], -1)
        z_gmm = torch.cat((mu, p, phi.unsqueeze(-1), eigen), dim=2).detach()
        z_gmm = self.from_gmm(z_gmm)  # apply identity transformation to g_j to get g^_j (i.e. no transformation in eq 7)
        zh_ = zh + z_gmm # see eq 7
        return zh_

    def compose_part_groups(self, tuples_id_to_gaussians, zc, gmms):
        """
        tuples_id_to_part_group: list of dicts, one per tuple. Each dict maps priming ids to the borrowed part group idx
            OR 
            tuples_id_to_gaussians: each dict maps priming ids to list of parts/gaussians borrowed. assume the borrowed gaussians are a partition of 16
        zc: [full_dataset_sz, m, dim]
        gmms: list of 4 x [B, 1,m,..]

        Returns: 
            - composed shapes zc: [len(dict), m, dim]
            - composed gaussians: 4 x [len(dict), 1,m, ..] where correct gaussians are borrowed
        """
        borrow_gaussians = True
        if borrow_gaussians:
            gmms_0 = [] #[len(dict), 1, m, dim]
            gmms_1 = []
            gmms_2 = []
            gmms_3 = []
            def sort_dict(dict):
                new_dict= {}
                for key, value in sorted(dict.items()): # Note the () after items!
                    new_dict[key] = value
                return new_dict
            
            composed_shapes_zc = -torch.ones((len(tuples_id_to_gaussians), zc.shape[1], zc.shape[2]))
            for i, tuple in enumerate(tuples_id_to_gaussians):
                curr_gmm0 = {} #stores gaussian id -> [dim,]
                curr_gmm1 = {} #stores gaussian id -> [dim,]
                curr_gmm2 = {} #stores gaussian id -> [dim,]
                curr_gmm3 = {} #stores gaussian id -> [dim,]
                for priming_shape_idx, gaussians_to_borrow in tuple.items():
                    # for this priming shape, copy the chosen group's parts to the output
                    for gaussian_id in gaussians_to_borrow:
                        composed_shapes_zc[i, gaussian_id] = zc[priming_shape_idx, gaussian_id]
                        curr_gmm0[gaussian_id] = gmms[0][priming_shape_idx, 0, gaussian_id] #[dim,]
                        curr_gmm1[gaussian_id] = gmms[1][priming_shape_idx, 0, gaussian_id] #[dim,]
                        curr_gmm2[gaussian_id] = gmms[2][priming_shape_idx, 0, gaussian_id].unsqueeze(0) #[dim,]
                        curr_gmm3[gaussian_id] = gmms[3][priming_shape_idx, 0, gaussian_id] #[dim,]
                gmms_0.append(torch.stack(list(sort_dict(curr_gmm0).values()),  dim=0))
                gmms_1.append(torch.stack(list(sort_dict(curr_gmm1).values()),  dim=0))
                gmms_2.append(torch.stack(list(sort_dict(curr_gmm2).values()),  dim=0))
                gmms_3.append(torch.stack(list(sort_dict(curr_gmm3).values()),  dim=0))
            gmms_0 = torch.stack(gmms_0).unsqueeze(1)
            gmms_1 = torch.stack(gmms_1).unsqueeze(1)
            gmms_2 = torch.stack(gmms_2).squeeze(dim=2).unsqueeze(1)
            gmms_3 = torch.stack(gmms_3).unsqueeze(1)
            return composed_shapes_zc.cuda(), [gmms_0, gmms_1,gmms_2,gmms_3]
        

        # else use part groups (OLD)
        gmms_0 = [] #[len(dict), 1, m, dim]
        gmms_1 = []
        gmms_2 = []
        gmms_3 = []

        # part_groups = {
        #     0: [0, 2, 3, 4, 8, 10, 11, 12], #body
        #     1: [7, 15], # outer wing
        #     2: [5, 13], # tail horiz stabilizer
        #     3: [6, 14], # tail vert wing
        #     4: [1, 9], #inner wing/engien
        # }
        def sort_dict(dict):
            new_dict= {}
            for key, value in sorted(dict.items()): # Note the () after items!
                new_dict[key] = value
            return new_dict
        composed_shapes_zc = -torch.ones((len(tuples_id_to_gaussians), zc.shape[1], zc.shape[2]))
        for i, tuple in enumerate(tuples_id_to_gaussians):
            curr_gmm0 = {} #stores gaussian id -> [dim,]
            curr_gmm1 = {} #stores gaussian id -> [dim,]
            curr_gmm2 = {} #stores gaussian id -> [dim,]
            curr_gmm3 = {} #stores gaussian id -> [dim,]
            for priming_shape_idx, part_group_to_borrow in tuple.items():
                # for this priming shape, copy the chosen group's parts to the output
                for gaussian_id in part_group_to_borrow:
                    composed_shapes_zc[i, gaussian_id] = zc[priming_shape_idx, gaussian_id]
                    curr_gmm0[gaussian_id] = gmms[0][priming_shape_idx, 0, gaussian_id] #[dim,]
                    curr_gmm1[gaussian_id] = gmms[1][priming_shape_idx, 0, gaussian_id] #[dim,]
                    curr_gmm2[gaussian_id] = gmms[2][priming_shape_idx, 0, gaussian_id].unsqueeze(0) #[dim,]
                    curr_gmm3[gaussian_id] = gmms[3][priming_shape_idx, 0, gaussian_id] #[dim,]
            gmms_0.append(torch.stack(list(sort_dict(curr_gmm0).values()),  dim=0))
            gmms_1.append(torch.stack(list(sort_dict(curr_gmm1).values()),  dim=0))
            gmms_2.append(torch.stack(list(sort_dict(curr_gmm2).values()),  dim=0))
            gmms_3.append(torch.stack(list(sort_dict(curr_gmm3).values()),  dim=0))
        gmms_0 = torch.stack(gmms_0).unsqueeze(1)
        gmms_1 = torch.stack(gmms_1).unsqueeze(1)
        gmms_2 = torch.stack(gmms_2).squeeze(dim=2).unsqueeze(1)
        gmms_3 = torch.stack(gmms_3).unsqueeze(1)
        return composed_shapes_zc.cuda(), [gmms_0, gmms_1,gmms_2,gmms_3]

    def merge_zh(self, zh, gmms, mask: Optional[T] = None, tuples_id_to_part_group=None) -> TNS:
        zh_ = self.merge_zh_step_a(zh, gmms) # z^_b part-level control (eq 7)
        if tuples_id_to_part_group:
            print("composing part groups")
            zh_, gmms = self.compose_part_groups(tuples_id_to_part_group, zh_, gmms)
        zh_, attn = self.mixing_network.forward_with_attention(zh_, mask=mask)  # z_c
        return zh_, attn, gmms

    def forward_b(self, x, zh, gmms, mask: Optional[T] = None) -> T:
        zh, _ = self.merge_zh(zh, gmms, mask)
        return self.occupancy_network(x, zh, mask)

    def forward_a(self, item: T):
        zh, z, gmms = self.get_embeddings(item)
        return zh, z, gmms

    def get_attention(self, x, item) -> TS:
        zh, z, gmms = self.forward_a(item)
        zh, _ = self.merge_zh(zh, gmms)
        return self.occupancy_network.forward_attention(x, zh)

    def forward(self, x, item: T) -> Tuple[T, T, TS, T]:
        zh, z, gmms = self.forward_a(item)
        return self.forward_b(x, zh, gmms), z, gmms, zh

    def forward_mid(self, x: T, zh: T) -> Tuple[T, TS]:
        zh, gmms = self.decomposition_control.forward_mid(zh)
        return self.forward_b(x, zh, gmms), gmms

    def get_random_embeddings(self, num_items: int):
        '''
        AT: returns randomly sampled vec z_a
        '''
        if self.dist is None:
            weights = self.z.weight.clone().detach()
            mean = weights.mean(0)
            weights = weights - mean[None, :]
            cov = torch.einsum('nd,nc->dc', weights, weights) / (weights.shape[0] - 1)
            self.dist = distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
        z_init = self.dist.sample((num_items,))
        return z_init

    def random_samples(self, num_items: int, output_dir='', tf_sample_dirname=''):
        z_init = self.get_random_embeddings(num_items) # random z_a, one per sample. [B, dim_z]
        zh, gmms = self.decomposition_control(z_init, output_dir, tf_sample_dirname)  # [B, m, dim_h]
        return zh, gmms

    def __init__(self, opt: Options):
        super(Spaghetti, self).__init__()
        self.device = opt.device
        self.opt = opt
        self.z = nn.Embedding(opt.dataset_size, opt.dim_z)
        print("dataset sz: ", opt.dataset_size)
        torch.nn.init.normal_(
            self.z.weight.data,
            0.0,
            1. / math.sqrt(opt.dim_z),
        )
        self.decomposition_control = DecompositionControl(opt)
        self.occupancy_network = OccupancyNetwork(opt)
        self.from_gmm = nn.Linear(sum(self.decomposition_control.split_shape), opt.dim_h)
        if opt.use_encoder:
            self.mixing_network = transformer.Transformer(opt.dim_h, opt.num_heads, opt.num_layers,
                                                              act=nnf.relu, norm_layer=nn.LayerNorm)
        else:
            self.mixing_network = transformer.DummyTransformer()
        self.dist = None
