from datasets import load_from_disk
import pathlib

def find_matching_index(l1,l2):
    inv_index = {element:index for index,element in enumerate(l1)}
    #return [(index,inv_index[element]) for index,element in enumerate(l2) if element in inv_index]
    return [index for index,element in enumerate(l2) if element in inv_index]
    
base_url = '/mnt/home/sho/ceph/20241008_cwan1/ceph/SDOML/AIA/'
channels = ['171','193','304','335']

paths = []
for channel in channels:
    url = base_url + channel + '/'
    channel_paths = sorted(list(pathlib.Path(url).rglob('*.npz')))
    paths.append(channel_paths)

paths_codes = [[],[],[],[]]
paths_codes[0] = [str(i)[70:80] for i in paths[0]]
paths_codes[1] = [str(i)[70:80] for i in paths[1]]
paths_codes[2] = [str(i)[70:80] for i in paths[2]]
paths_codes[3] = [str(i)[70:80] for i in paths[3]]

common_codes = sorted(set(paths_codes[0]).intersection(paths_codes[1]).intersection(paths_codes[2]).intersection(paths_codes[3]))

valid_171 = find_matching_index(common_codes,paths_codes[0])
valid_193 = find_matching_index(common_codes,paths_codes[1])
valid_304 = find_matching_index(common_codes,paths_codes[2])
valid_335 = find_matching_index(common_codes,paths_codes[3])

ds_171 = load_from_disk("/mnt/home/sho/ceph/20241008_cwan1/ceph/datasets/AIA/171")
ds_171 = ds_171.with_format("numpy")

ds_193 = load_from_disk("/mnt/home/sho/ceph/20241008_cwan1/ceph/datasets/AIA/193")
ds_193 = ds_193.with_format("numpy")

ds_304 = load_from_disk("/mnt/home/sho/ceph/20241008_cwan1/ceph/datasets/AIA/304")
ds_304 = ds_304.with_format("numpy")

ds_335 = load_from_disk("/mnt/home/sho/ceph/20241008_cwan1/ceph/datasets/AIA/335")
ds_335 = ds_335.with_format("numpy")

ds_171_valid = ds_171.select(valid_171)
ds_193_valid = ds_193.select(valid_193)
ds_304_valid = ds_304.select(valid_304)
ds_335_valid = ds_335.select(valid_335)

ds_171_valid.save_to_disk('/mnt/home/manand/ceph/171')
ds_193_valid.save_to_disk('/mnt/home/manand/ceph/193')
ds_304_valid.save_to_disk('/mnt/home/manand/ceph/304')
ds_335_valid.save_to_disk('/mnt/home/manand/ceph/335')
