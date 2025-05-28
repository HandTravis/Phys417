import h5py
import matplotlib.pyplot as plt

dataset_path = './lab8-9/GW2_Andy.h5'
datasets = {}
# Open the file in read mode
with h5py.File(dataset_path, 'r') as f:
    # List all groups/datasets
    print("Keys: %s" % list(f.keys()))
    
    for key in f.keys():
        dataset = f[key][:]  # replace 'dataset_name' with actual key
        datasets[key] = dataset
        # Check its shape, type, etc.
        print(dataset.shape)
        print(dataset.dtype)
        print()

plt.figure(figsize=(12, 4))
glitch_0 = datasets['glitch'][0]
bbh_0 = datasets['binaryblackhole'][0]
ccsn_0 = datasets['ccsn'][0]
background_0 = datasets['background'][0]
# plt.plot(glitch_0[0], label='Glitch sensor 1')
# plt.plot(glitch_0[1], label='Glitch sensor 2')
plt.plot(bbh_0[0], label='BBH Sensor 1')
plt.plot(bbh_0[1], label='BBH Sensor 2')
# plt.plot(ccsn_0[0], label='CCSN Sensor 1')
# plt.plot(ccsn_0[1], label='CCSN Sensor 2')
# plt.plot(background_0[0], label='Background Sensor 1')
# plt.plot(background_0[1], label='Background Sensor 2')
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


    
