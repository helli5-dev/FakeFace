from numpy import load
from numpy import savez_compressed
from numpy import asarray

unparted_array = load('celeba128.npz')
unparted_array = unparted_array['arr_0']
print(unparted_array.shape)

part1 = unparted_array[:10000]
part2 = unparted_array[10000:20000]
part3 = unparted_array[20000:30000]
part4 = unparted_array[30000:40000]
part5 = unparted_array[40000:]

savez_compressed("dataset_part1.npz", asarray(part1))
savez_compressed("dataset_part2.npz", asarray(part2))
savez_compressed("dataset_part3.npz", asarray(part3))
savez_compressed("dataset_part4.npz", asarray(part4))
savez_compressed("dataset_part5.npz", asarray(part5))