import numpy as np
from floaders import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

cut = 22.5

set = shape_dataset(use='all', std=0)

# angles = [*range(0, 360, 45)]
# angles = None
angles = [48.54946536, 90.02852279, 129.56504831, 164.66412299, 210.77101218, 257.82223983, 310.82768749, 345.781062]

random_state = 1228

if angles is None:
    n_clusters = 8
    centers = None
    for i in range(len(set)):
        dir = set.X[i]
        meta = np.genfromtxt(dir + '/rendering_metadata.txt', delimiter=' ')
        ang = meta[:, 0] * np.pi / 180
        X = np.transpose([np.cos(ang), np.sin(ang)])
        km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
        centers = km.cluster_centers_ if centers is None else np.append(centers, km.cluster_centers_, axis=0)
    centers = np.transpose(np.divide(np.transpose(centers), np.sum(centers ** 2, axis=1) ** (1./2)))
    km = KMeans(n_clusters=n_clusters, random_state=random_state).fit(centers)
    cnt = km.cluster_centers_
    ac = np.arccos(cnt[:, 0])
    angles = ac + 2 * (cnt[:, 1] < 0).astype(float) * (np.pi - ac)
    angles *= 180 / np.pi
    angles.sort()
    print(angles)
else:
    n_clusters = len(angles)

diffs = np.zeros((len(set), n_clusters))
for i in range(len(set)):
    dir = set.X[i]
    meta = np.genfromtxt(dir + '/rendering_metadata.txt', delimiter=' ')
    diff = np.abs(np.subtract(np.transpose(angles), np.expand_dims(meta[:, 0], axis=1)))
    diff2 = np.abs(np.subtract(np.transpose(angles) - 360, np.expand_dims(meta[:, 0], axis=1)))
    dmin = diff.min(axis=0)
    dmin2 = diff2.min(axis=0)
    diffs[i, :] = np.minimum(dmin, dmin2)

diffcuts = diffs < cut
diffcut = np.prod(diffcuts, axis=1)

print('{0} of {1} samples have all views within cutoff of {2} degrees ({3:.2f}%)'.format(np.sum(diffcut),
                                                                                         len(diffcut), cut,
                                                                                         100 * np.mean(diffcut)))
print('{0} of {1} views are within cutoff of {2} degrees ({3:.2f}%)'.format(np.sum(diffcuts),
                                                                            np.product(diffcuts.shape), cut,
                                                                            100 * np.mean(diffcuts)))

xaxis = 'Distance'
plt.hist(diffs.flatten(), bins=25, range=(0, 50), density=True)
plt.ylim(0, 0.2)
plt.xlabel(xaxis)
plt.savefig('./plots/shape_angle_hist.png')
plt.clf()
plt.close()

for i in range(n_clusters):
    plt.hist(diffs[:, i], bins=25, range=(0, 50), density=True)
    plt.ylim(0, 0.2)
    plt.title('{:.2f}'.format(angles[i]))
    plt.xlabel(xaxis)
    plt.savefig('./plots/shape_angle_hist_{0}.png'.format(angles[i]))
    plt.clf()
    plt.close()
