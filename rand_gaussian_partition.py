

import json
import random

test_tuples_dir = "assets/checkpoints/spaghetti_airplanes/rand_unconditional/codes/11-13-2024--00-41-13/"
test_tuples_filename = "retrieval_random_close_variety_zb_8k_info_test.json"
num_priming_shapes = 5


test_tuples = json.load(open(test_tuples_dir+"/"+test_tuples_filename))
target_shape_ids = test_tuples.keys()
def random_partition(lst, n):
    # Shuffle the list
    random.shuffle(lst)

    # Initialize the output lists
    partitions = []

    # Generate random lengths for the partitions
    lengths = sorted(random.sample(range(0, len(lst)), n - 1))
    lengths.append(len(lst))

    # Use the lengths to slice the list and create the partitions
    start = 0
    for end in lengths:
        partitions.append(lst[start:end])
        start = end

    return partitions

random_output_tuples = []

## temp: for each slot sample without replacement
for _, input_priming_id_to_gaussians in test_tuples.items():
    # first randomly select 8 with replacement
    part_groups = random.choices(list(range(8)), k=8)
    print(part_groups)
    # make "partition" ie. random sublists
    partition = random_partition(part_groups, n=num_priming_shapes)
    # add symmetric gaussians (offset each index by +8)
    output_priming_id_to_gaussians = {}
    for priming_id, part_group in zip(input_priming_id_to_gaussians.keys(), partition):
        symmetrized_part_group = part_group.copy()
        for gaussian in part_group:
            symmetrized_part_group.append(gaussian+8)
        output_priming_id_to_gaussians[priming_id] = symmetrized_part_group
    random_output_tuples.append(output_priming_id_to_gaussians)
##

## without replacement
# for _, input_priming_id_to_gaussians in test_tuples.items():
#     # randomly partition 16 into num_priming_shapes parts (potentially empty)
#     # first partition 8
#     partition = random_partition(list(range(8)), n=num_priming_shapes)
#     print(partition)
#     # add symmetric gaussians (offset each index by +8)
#     output_priming_id_to_gaussians = {}
#     for priming_id, part_group in zip(input_priming_id_to_gaussians.keys(), partition):
#         symmetrized_part_group = part_group.copy()
#         for gaussian in part_group:
#             symmetrized_part_group.append(gaussian+8)
#         output_priming_id_to_gaussians[priming_id] = symmetrized_part_group
#     random_output_tuples.append(output_priming_id_to_gaussians)


for priming_id_to_gaussians in random_output_tuples:
    print("====")
    for priming_id, part_group in priming_id_to_gaussians.items():
        print(f"{priming_id}: {part_group}")

# Serializing json
json_object = json.dumps(random_output_tuples, indent=4)
# Writing to sample.json
with open(f"{test_tuples_dir}/random_symmetric_gaussian_selection.json", "w") as outfile:
    outfile.write(json_object)