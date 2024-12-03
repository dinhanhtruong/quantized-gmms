import os
import random
from custom_types import *
import options
from options import Options, recon_sample_offset
from utils import train_utils, mcubes_meshing, files_utils, mesh_utils
from models.occ_gmm import Spaghetti
from models import models_utils
import json

def get_gm_support(gm, x):
    dim = x.shape[-1] #3
    mu, p, phi, eigen = gm
    sigma_det = eigen.prod(-1)
    eigen_inv = 1 / eigen
    sigma_inverse = torch.matmul(p.transpose(3, 4), p * eigen_inv[:, :, :, :, None]).squeeze(1)
    phi = torch.softmax(phi, dim=2)
    const_1 = phi / torch.sqrt((2 * np.pi) ** dim * sigma_det)
    distance = x[:, :, None, :] - mu
    mahalanobis_distance = - .5 * torch.einsum('bngd,bgdc,bngc->bng', distance, sigma_inverse, distance)
    const_2, _ = mahalanobis_distance.max(dim=2)  # for numeric stability
    mahalanobis_distance -= const_2[:, :, None]
    support = const_1 * torch.exp(mahalanobis_distance)
    return support, const_2


def gm_log_likelihood_loss(gms: TS, x: T, reduction: str = "mean") -> Union[T, Tuple[T, TS]]:

    # batch_size, num_points, dim = x.shape
    support, const = get_gm_support(gms, x)

    # probs = torch.log(support.sum(dim=2)) + const
    # if reduction == 'none':
    #     likelihood = probs.sum(-1)
    #     loss = - likelihood / num_points
    # else:
    #     likelihood = probs.sum()
    #     loss = - likelihood / (probs.shape[0] * probs.shape[1])

    return support


class Inference:

    def get_occ_fun(self, z: T):

        def forward(x: T) -> T:
            nonlocal z
            x = x.unsqueeze(0)
            out = self.model.occupancy_network(x, z)[0, :]
            out = 2 * out.sigmoid_() - 1
            return out

        if z.dim() == 2:
            z = z.unsqueeze(0)
        return forward

    @models_utils.torch_no_grad
    def get_mesh(self, z: T, res: int) -> Optional[T_Mesh]:
        mesh = self.meshing.occ_meshing(self.get_occ_fun(z), res=res)
        return mesh
    
    def load_mesh_names(self, mesh_names_path):
        # TEMP: load train set raw mesh names
        if not self.raw_mesh_names:
            data = json.load(open(mesh_names_path))
            self.raw_mesh_names = data["ShapeNetV2"]["02691156"]

    def get_top_gaussian_colors(self, sample_points, gmms, attn_weights=None):
        '''
        Retrieves vertex colors for a single shape

        Args
            - sample_points: [v, 3]
            - gmms: collection of gmm params:
                - mu=centroids  [m,3]
                - p=factorized cov matrices [m, 3,3]
                - phi=mixing weights [m]
                - eigenvals  [m,3]
            - attn_weights: [m,]

        returns: int RGB vertex colors[v, 3]
        '''
        sample_points = sample_points.cuda()
        means, p, mix_weights, eigenvals = gmms
        num_parts = mix_weights.shape[0]
        # assert num_parts == 16
        gmm_sample_vals = []
        print("v: ", sample_points.shape)

        # reshape to [1, ...] for batch dim compatibility
        sample_points = sample_points.unsqueeze(0)
        # reshape to original with batch dim 1
        gmms = (means.view(1,1,-1,3), p.view(1,1,-1,3,3), mix_weights.view(1,1,-1), eigenvals.view(1,1,-1,3))

        gmm_sample_vals = get_gm_support(gmms, sample_points)[0][0] #ignore aux 2nd output, get 1st item w batch sz=1. shape [v,m]
        # get argmax gaussian for each vert sample
        gaussian_labels = torch.argmax(gmm_sample_vals, dim=1) #[v]
        print("gaussian labels: ", gaussian_labels)
        print("num unique: ", torch.unique(gaussian_labels))
        
        if attn_weights is not None:
            print("using attention weights for coloring")
            gmm_color_weights = attn_weights.unsqueeze(1)  # [m, 1]
        else:
            gmm_color_weights = (torch.arange(16)/15).view(16,1).cuda()
        red = torch.tensor([
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0],
            [255,0,0]
        ]).cuda()
        green = torch.tensor([
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0],
            [0,255,0]
        ]).cuda()
        # interpolate b/t red and green (red = highest weight)
        gmm_colors = gmm_color_weights*red + (1-gmm_color_weights)*green
        
        # gmm_colors = torch.tensor([
        #     [150,150,150],
        #     [127,0,0],
        #     [150,150,150],
        #     [150,150,150],
        #     [150,150,150],
        #     [255,165,0],
        #     [255,255,0],
        #     [150,150,150],
        #     [150,150,150],
        #     [0,0,255],
        #     [150,150,150],
        #     [150,150,150],
        #     [150,150,150],
        #     [144,240,144],
        #     [150,150,150],
        #     [150,150,150]
        # ]).cuda()

        # # part group colors
        # gmm_colors = torch.tensor([
        #     [127,0,0],
        #     [144,240,144],
        #     [127,0,0],
        #     [127,0,0],
        #     [127,0,0],
        #     [255,255,0],
        #     [0,0,255],
        #     [255,165,0],
        #     [127,0,0],
        #     [144,240,144],
        #     [127,0,0],
        #     [127,0,0],
        #     [127,0,0],
        #     [255,255,0],
        #     [0,0,255],
        #     [255,165,0]
        # ]).cuda()
        
        # high contrast colors
        gmm_colors = torch.tensor([
            [47,80,80],
            [127,0,0],
            [25,25,112],
            [0,100,0],
            [255,0,0],
            [255,165,0],
            [255,255,0],
            [0,255,0],
            [0,255,255],
            [0,0,255],
            [255,0,255],
            [30,144,255],
            [220,160,220],
            [144,240,144],
            [255,20,150],
            [255,220,185]
        ]).cuda()
        
        return gmm_colors[gaussian_labels]


    def plot_occ(self, z: Union[T, TS], z_base, gmms: Optional[TS], fixed_items: T,
                 folder_name: str, res=200, verbose=True, from_quantized=False, tf_sample_dirname='', attn_weights=None):
        self.load_mesh_names(f'{self.opt.cp_folder}/shapenet_airplanes_train.json')
        
        means, eigenvecs, mix_weights, eigenvals = gmms
        
        for i, spaghetti_shape_idx in enumerate(fixed_items):
            spaghetti_shape_idx = spaghetti_shape_idx.item()
            print("(spag) shape idx: ", spaghetti_shape_idx)
            # if i == 10:
            #     exit()
            mesh = self.get_mesh(z[spaghetti_shape_idx], res)  # mcubes.  tuple of (V,F)
            # name = f'{fixed_items[i]:04d}' # OLD naming: use latent vec ID
            if options.use_quantized:
                if tf_sample_dirname:
                    name = f'sample_{i+options.recon_sample_offset}' # overwrite name; ignore shapenet IDs
                elif attn_weights is not None:
                    name = self.raw_mesh_names[spaghetti_shape_idx] # use raw shapenet mesh name
                    name += '_attn_colored'
                elif from_quantized:
                    name = self.raw_mesh_names[spaghetti_shape_idx] # use raw shapenet mesh name
                    name += '_quantized'
            else:
                name = str(spaghetti_shape_idx+options.recon_sample_offset)
                
            if mesh is not None:
                # temp: color verts based on gaussian maximimizing likelihood
                vert_colors = self.get_top_gaussian_colors(
                    mesh[0], 
                    (means[spaghetti_shape_idx,0], eigenvecs[spaghetti_shape_idx,0], mix_weights[spaghetti_shape_idx,0], eigenvals[spaghetti_shape_idx,0]),
                    attn_weights=None if attn_weights is None else attn_weights[i]
                )
                if tf_sample_dirname:
                    # vert_colors = torch.randint(0, 255, (mesh[0].shape[0], 3)) #[v,3]
                    files_utils.export_mesh(mesh, f'{self.opt.cp_folder}/{folder_name}/occ/{tf_sample_dirname}/{name}', vert_colors)
                else:
                    files_utils.export_mesh(mesh, f'{self.opt.cp_folder}/{folder_name}/occ/{name}', vert_colors) # obj
                # files_utils.save_pickle(z_base[i].detach().cpu(), f'{self.opt.cp_folder}/{folder_name}/occ/{name}')
                if gmms is not None:
                    pass
                    files_utils.export_gmm(gmms, spaghetti_shape_idx, f'{self.opt.cp_folder}/{folder_name}/gmms/{name}')
            # if verbose:
            print(f'done {i + 1:d}/{len(z):d}')

    def load_file(self, info_path, disclude: Optional[List[int]] = None):
        info = files_utils.load_pickle(''.join(info_path))
        keys = list(info['ids'].keys())
        items = map(lambda x: int(x.split('_')[1]) if type(x) is str else x, keys)
        items = torch.tensor(list(items), dtype=torch.int64, device=self.device)
        zh, _, gmms_sanity, _ = self.model.get_embeddings(items)
        gmms = [item for item in info['gmm']]
        zh_ = []
        split = []
        gmm_mask = torch.ones(gmms[0].shape[2], dtype=torch.bool)
        counter = 0
        # gmms_ = [[] for _ in range(len(gmms))]
        for i, key in enumerate(keys):
            gaussian_inds = info['ids'][key]
            if disclude is not None:
                for j in range(len(gaussian_inds)):
                    gmm_mask[j + counter] = gaussian_inds[j] not in disclude
                counter += len(gaussian_inds)
                gaussian_inds = [ind for ind in gaussian_inds if ind not in disclude]
                info['ids'][key] = gaussian_inds
            gaussian_inds = torch.tensor(gaussian_inds, dtype=torch.int64)
            zh_.append(zh[i, gaussian_inds])
            split.append(len(split) + torch.ones(len(info['ids'][key]), dtype=torch.int64, device=self.device))
        zh_ = torch.cat(zh_, dim=0).unsqueeze(0).to(self.device)
        gmms = [item[:, :, gmm_mask].to(self.device) for item in info['gmm']]
        return zh_, gmms, split, info['ids']

    @models_utils.torch_no_grad
    def get_z_from_file(self, info_path):
        zh_, gmms, split, _ = self.load_file(info_path)
        zh_ = self.model.merge_zh_step_a(zh_, [gmms])
        zh, _ = self.model.affine_transformer.forward_with_attention(zh_)
        # gmms_ = [torch.cat(item, dim=1).unsqueeze(0) for item in gmms_]
        # zh, _ = self.model.merge_zh(zh_, [gmms])
        return zh, zh_, gmms, torch.cat(split)

    def plot_from_info(self, info_path, res):
        zh, zh_, gmms, split = self.get_z_from_file(info_path)
        mesh = self.get_mesh(zh[0], res, gmms)
        if mesh is not None:
            attention = self.get_attention_faces(mesh, zh, fixed_z=split)
        else:
            attention = None
        return mesh, attention

    @staticmethod
    def combine_and_pad(zh_a: T, zh_b: T) -> Tuple[T, TN]:
        if zh_a.shape[1] == zh_b.shape[1]:
            mask = None
        else:
            pad_length = max(zh_a.shape[1], zh_b.shape[1])
            mask = torch.zeros(2, pad_length, device=zh_a.device, dtype=torch.bool)
            padding = torch.zeros(1, abs(zh_a.shape[1] - zh_b.shape[1]), zh_a.shape[-1], device=zh_a.device)
            if zh_a.shape[1] > zh_b.shape[1]:
                mask[1, zh_b.shape[1]:] = True
                zh_b = torch.cat((zh_b, padding), dim=1)
            else:
                mask[0, zh_a.shape[1]:] = True
                zh_a = torch.cat((zh_a, padding), dim=1)
        return torch.cat((zh_a, zh_b), dim=0), mask

    @staticmethod
    def get_intersection_z(z_a: T, z_b: T) -> T:
        diff = (z_a[0, :, None, :] - z_b[0, None]).abs().sum(-1)
        diff_a = diff.min(1)[0].lt(.1)
        diff_b = diff.min(0)[0].lt(.1)
        if diff_a.shape[0] != diff_b.shape[0]:
            padding = torch.zeros(abs(diff_a.shape[0] - diff_b.shape[0]), device=z_a.device, dtype=torch.bool)
            if diff_a.shape[0] > diff_b.shape[0]:
                diff_b = torch.cat((diff_b, padding))
            else:
                diff_a = torch.cat((diff_a, padding))
        return torch.cat((diff_a, diff_b))

    def get_attention_points(self, vs: T, zh: T, mask: TN = None, alpha: TN = None):
        vs = vs.unsqueeze(0)
        attention = self.model.occupancy_network.forward_attention(vs, zh, mask=mask, alpha=alpha)
        attention = torch.stack(attention, 0).mean(0).mean(-1)
        attention = attention.permute(1, 0, 2).reshape(attention.shape[1], -1)
        attention_max = attention.argmax(-1)
        return attention_max

    @models_utils.torch_no_grad
    def get_attention_faces(self, mesh: T_Mesh, zh: T, mask: TN = None, fixed_z: TN = None, alpha: TN = None):
        coords = mesh[0][mesh[1]].mean(1).to(zh.device)

        attention_max = self.get_attention_points(coords, zh, mask, alpha)
        if fixed_z is not None:
            attention_select = fixed_z[attention_max].cpu()
        else:
            attention_select = attention_max
        return attention_select

    @models_utils.torch_no_grad
    def plot_folder(self, *folders, res: int = 256):
        logger = train_utils.Logger()
        for folder in folders:
            paths = files_utils.collect(folder, '.pkl')
            logger.start(len(paths))
            for path in paths:
                name = path[1]
                out_path = f"{self.opt.cp_folder}/from_ui/{name}"
                mesh, colors = self.plot_from_info(path, res)
                if mesh is not None:
                    files_utils.export_mesh(mesh, out_path)
                    files_utils.export_list(colors.tolist(), f"{out_path}_faces")
                logger.reset_iter()
            logger.stop()

    def get_zh_from_idx(self, items: T):
        zh, _, gmms, __ = self.model.get_embeddings(items.to(self.device))
        zh, attn_b = self.model.merge_zh(zh, gmms)
        return zh, gmms

    @property
    def device(self):
        return self.opt.device

    def get_new_ids(self, folder_name, nums_sample):
        names = [int(path[1]) for path in files_utils.collect(f'{self.opt.cp_folder}/{folder_name}/occ/', '.obj')]
        ids = torch.arange(nums_sample)
        if len(names) == 0:
            return ids + self.opt.dataset_size
        return ids + max(max(names) + 1, self.opt.dataset_size)

    @models_utils.torch_no_grad
    ##########################################
    def random_plot(self, folder_name: str, nums_sample, res=200, verbose=False, tf_sample_dirname=''):
        '''
        Saves randomly sampled mesh (and GMMs)
        '''
        print("rand shape")
        zh_base, gmms = self.model.random_samples(nums_sample, folder_name, tf_sample_dirname) # get surface vecs s_j and GMM params
        centroids, factorized_cov, mixing_weights, eigenvals = gmms
        



        zh, attn_b = self.model.merge_zh(zh_base, gmms)  # z_c after applying mixing net 
        # numbers = self.get_new_ids(folder_name, nums_sample)

        sample_numbers = torch.arange(nums_sample)


        self.plot_occ(zh, zh_base, gmms, sample_numbers, folder_name, verbose=verbose, res=res)
    ###########################################

    def string_to_int_keys(self, tuples_id_to_part_group):
        output = []
        for tuple in tuples_id_to_part_group:
            curr_dict = {int(k): v for k,v in tuple.items()}
            output.append(curr_dict)
        return output
    
    def get_canonical_part_group_order(self, tuples_id_to_part_group):
        output = []
        for tuple in tuples_id_to_part_group:
            curr_dict = {}
            for canonical_v, k  in enumerate(tuple.keys()):
                curr_dict[k] = canonical_v
            output.append(curr_dict)
        return output
    
    def get_random_part_group_order(self, tuples_id_to_part_group):
        output = []
        for tuple in tuples_id_to_part_group:
            rand_part_groups = list(range(5)) 
            random.shuffle(rand_part_groups)
            curr_dict = {}
            for i, k  in enumerate(tuple.keys()):
                # randomly remove from set
                curr_dict[k] = rand_part_groups[i]
            output.append(curr_dict)
        return output

    @models_utils.torch_no_grad
    def plot(self, folder_name: str, nums_sample: int, verbose=False, res: int = 200, tf_sample_dirname='', attn_weights_path=''):
        '''
        Saves reconstructions of training meshes
        '''
        attn_weights = None
        tuples_id_to_part_group= None
        if attn_weights_path:
            assert not tf_sample_dirname # want standard quantized reconstruction process
            shape_samples = os.path.basename(attn_weights_path)[:-3].split("_")[-1]
            print(shape_samples)
            shape_samples = shape_samples.split("-")
            shape_samples = torch.tensor([int(x) for x in shape_samples]) # ints
            print("attn weight context shapes: ", shape_samples)
            attn_weights = torch.load(attn_weights_path) # [total_n_parts,]
            # linearly rescale attention weights to [0,1] range
            attn_weights = (attn_weights - torch.min(attn_weights))/(torch.max(attn_weights) - torch.min(attn_weights))
            attn_weights = attn_weights.view(shape_samples.shape[0], -1) # [n_priming_shapes, n_parts]
            print("scaled attn: ", attn_weights)
        elif self.model.opt.dataset_size < nums_sample:
            print("using ALL train data")
            shape_samples = torch.arange(self.model.opt.dataset_size)
        else:
            # print('using rand train subset (if no TF samples specified)')
            # shape_samples = torch.randint(low=0, high=self.opt.dataset_size, size=(nums_sample,))
            print('using ordered train subset (if no TF samples specified)')
            shape_samples = torch.arange(nums_sample)
        if tf_sample_dirname:
            print("using TF samples from ", tf_sample_dirname)
        
            tuples_id_to_part_group = json.load(open(f"assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/{tf_sample_dirname}/tuples_id_to_part_group.json"))
            # tuples_id_to_part_group = json.load(open(f"assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/{tf_sample_dirname}/tuple_0_converted.json"))
            # tuples_id_to_part_group = json.load(open(f"assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/{tf_sample_dirname}/random_symmetric_gaussian_selection.json"))
            tuples_id_to_part_group = self.string_to_int_keys(tuples_id_to_part_group)

            # UNCOMMENT FOR "GT" ORDERING (01234)
            # tuples_id_to_part_group = self.get_canonical_part_group_order(tuples_id_to_part_group)
            # tuples_id_to_part_group = self.get_random_part_group_order(tuples_id_to_part_group)

            print(f"tuples_id_to_part_group: ")
            for tuple in tuples_id_to_part_group:
                print(f"\t{tuple}")
            # tuples_id_to_part_group = [
            #     {
            #         9635: 0,
            #         3372: 1,
            #         3953: 2,
            #         5942: 3,
            #         4433: 4,
            #     },
            #     {
            #         7792: 0,
            #         1242: 1,
            #         5125: 2,
            #         2042: 3,
            #         7976: 4,
            #     }
            # ]
                
        zh_base, _, gmms = self.model.get_embeddings(shape_samples.to(self.device), folder_name, tf_sample_dirname) # NOTE: quantized code overrides internally
        # torch.save(gmms, f'assets/checkpoints/spaghetti_airplanes/{folder_name}/gmms.pt')
        # torch.save(zh_base, f'assets/checkpoints/spaghetti_airplanes/{folder_name}/zh_base.pt')

        if options.use_salad_data:
            print("using SALAD-generated priming shapes")
            tuple_indices = torch.load(f'assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/salad_val_indices.pt')
            tuple_index = tuple_indices[0]
            salad_zh_base = torch.load(f'assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/salad_intrinsic.pt') # [B, k=5, m=16, dim=512]
            zh_base = salad_zh_base[tuple_index] # [k=5, m=16, dim=512]

            salad_gmm = torch.load(f'assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/salad_gmm.pt')
            b, gp, g, _ = salad_gmm[0].shape
            gmms = [item.view(b, gp * g, *item.shape[3:]) for item in salad_gmm]

        zh, attn_b, gmms = self.model.merge_zh(zh_base, gmms, tuples_id_to_part_group=tuples_id_to_part_group)  # z^c [B, m, dim=512]. index into gmms if necessary


        # save mesh names
        from_quantized = False
        if not attn_weights_path: 
            if not os.path.exists(f'assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/mesh_ids.npy'):
                np.save(f'assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/mesh_ids.npy', shape_samples.detach().cpu().numpy())
            # else:
            #     print('using existing mesh names')
            #     from_quantized = True
            #     shape_samples = np.load(f'assets/checkpoints/spaghetti_airplanes/{folder_name}/codes/mesh_ids.npy')

        
        # TEMPP: uncomment for reconstructing gt of specified eval tuples (via index into saved codebook_indices)
        # shape_samples = torch.tensor([int(x) for x in [3, 4, 11, 13,15]])

        # only reconstruct meshes for first n samples
        shape_samples = shape_samples[:nums_sample]

        # # temp: select shape indices for reconstruction from tuples (indices must match ordering of latents loaded)
        tuple_to_reconstruct = [
            "1137",
            "15",
            "248",
            "35",
            "2164"
        ]
        shape_samples = torch.tensor([int(x) for x in tuple_to_reconstruct], dtype=torch.int)

        # TEMP: massive recon
        # shape_samples = torch.arange(nums_sample)
        self.plot_occ(zh, zh_base, gmms, shape_samples, folder_name, verbose=True, res=res, from_quantized=from_quantized, tf_sample_dirname=tf_sample_dirname, attn_weights=attn_weights)

    def get_mesh_from_mid(self, gmm, included: T, res: int) -> Optional[T_Mesh]:
        if self.mid is None:
            return None
        gmm = [elem.to(self.device) for elem in gmm]
        included = included.to(device=self.device)
        mid_ = self.mid[included[:, 0], included[:, 1]].unsqueeze(0)
        zh = self.model.merge_zh(mid_, gmm)[0]
        mesh = self.get_mesh(zh[0], res)
        return mesh

    def set_items(self, items: T):
        self.mid = items.to(self.device)

    def __init__(self, opt: Options):
        self.opt = opt
        model: Tuple[Spaghetti, Options] = train_utils.model_lc(opt)
        self.model, self.opt = model
        self.model.eval()
        self.mid: Optional[T] = None
        self.gmms: Optional[TN] = None
        self.meshing = mcubes_meshing.MarchingCubesMeshing(self.device, scale=1.)
        self.raw_mesh_names = None

