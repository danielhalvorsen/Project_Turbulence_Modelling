import numpy as np

dataset = np.load('particleCoord_8000.npy')

varX_list = []
varY_list = []
varZ_list = []


datasetshape = np.shape(dataset)
timesteps = datasetshape[0]

for i in range(timesteps):
    varX_list.append(np.var(dataset[i][0,:]))
    varY_list.append(np.var(dataset[i][1,:]))
    varZ_list.append(np.var(dataset[i][2,:]))
    print(i,flush=True)

np.save('VarX_list.npy',np.array(varX_list))
np.save('VarY_list.npy',np.array(varY_list))
np.save('VarZ_list.npy',np.array(varZ_list))