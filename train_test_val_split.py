from datasets import load_from_disk
#import pathlib

ds_171 = load_from_disk('/mnt/home/manand/ceph/171').with_format("numpy")
ds_193 = load_from_disk('/mnt/home/manand/ceph/193').with_format("numpy")
ds_304 = load_from_disk('/mnt/home/manand/ceph/304').with_format("numpy")
ds_335 = load_from_disk('/mnt/home/manand/ceph/335').with_format("numpy")

test_indices = [i for i in range(378307,462213+1)]
val_indices = [i for i in range(339042,378306+1)] + [j for j in range(462214,504545+1)]
train_indices = [i for i in range(0,339042)] + [j for j in range(504546,len(ds_171))]

print('Saving Train')
ds_171_valid = ds_171.select(train_indices)
ds_193_valid = ds_193.select(train_indices)
ds_304_valid = ds_304.select(train_indices)
ds_335_valid = ds_335.select(train_indices)

ds_171_valid.save_to_disk('/mnt/home/manand/ceph/171_train')
ds_193_valid.save_to_disk('/mnt/home/manand/ceph/193_train')
ds_304_valid.save_to_disk('/mnt/home/manand/ceph/304_train')
ds_335_valid.save_to_disk('/mnt/home/manand/ceph/335_train')

print('Saving Test')
ds_171_valid = ds_171.select(test_indices)
ds_193_valid = ds_193.select(test_indices)
ds_304_valid = ds_304.select(test_indices)
ds_335_valid = ds_335.select(test_indices)

ds_171_valid.save_to_disk('/mnt/home/manand/ceph/171_test')
ds_193_valid.save_to_disk('/mnt/home/manand/ceph/193_test')
ds_304_valid.save_to_disk('/mnt/home/manand/ceph/304_test')
ds_335_valid.save_to_disk('/mnt/home/manand/ceph/335_test')

print('Saving Val')
ds_171_valid = ds_171.select(val_indices)
ds_193_valid = ds_193.select(val_indices)
ds_304_valid = ds_304.select(val_indices)
ds_335_valid = ds_335.select(val_indices)

ds_171_valid.save_to_disk('/mnt/home/manand/ceph/171_val')
ds_193_valid.save_to_disk('/mnt/home/manand/ceph/193_val')
ds_304_valid.save_to_disk('/mnt/home/manand/ceph/304_val')
ds_335_valid.save_to_disk('/mnt/home/manand/ceph/335_val')

'''n=len(ds_171)
for index in range(0,n,1000):
    print(index,ds_171[index]['image']['date'])


test_start = range(378000,379000)
print('2015 Start, test start')
for i in test_start:
    print(i,ds_171[i]['image']['date'])
#378307

test_start = range(462000,463000)
print('2015 End, test end')
for i in test_start:
    print(i,ds_171[i]['image']['date'])
#462213

test_start = range(339000,340000)
print('July 2014 Start, val start')
for i in test_start:
    print(i,ds_171[i]['image']['date'])
#339042

test_start = range(378000,379000)
print('Dec 2014 End, val_1 end')
for i in test_start:
    print(i,ds_171[i]['image']['date'])
#378306

test_start = range(462000,463000)
print('Jan 2016 Start, val_2 start')
for i in test_start:
    print(i,ds_171[i]['image']['date'])
#462214

test_start = range(504000,505000)
print('June 2016 end, val end')
for i in test_start:
    print(i,ds_171[i]['image']['date'])
#504545''' 