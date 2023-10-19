import numpy as np
from floaders import *
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

sh = 341  # 1, 85, 171, 255, 341
if sh == 1:
    c0 = [*range(348, 352), 32, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46, 58, 259, 209, 220, 312, 1, 12, 30,
          61, 83, 89, 113, 145, 232, 235, 272, 281, 300, 311, 322, 34, 49, 202, 212, 217, 233, 269, 270, 271, 287, 305,
          330, 191, 226, 228, 273, 298, 6, 82, 92, 225, 319, 40, 47, 75, 77, 78, 117, 248, 267, 123, 208, 67, 219, 304,
          218, *range(335, 340), 13, 25, 35, 38, 86, 107, 121, 170, 172, 174, 204, 206, 207, 210, 211, 139, 294, 129,
          130, 131, 132, 133, 134, 135, 136, 137, 138, 90, 164, 177, 180, 181, 182, 183, 184, 195, 198, 199, 200, 21,
          33, 120, 122, 152, 153, 154, 155, 95, 252, 42, 240, 106, 323, 87, 246, 14, 186, 16, 274, 22, 296, 282, 8, 241,
          277, 24, 27, 243, 234, 276, 283, 105, 229, 316, 93, 140, 262, 260, 279, 54, 146, 318, 265, 148, 150, 151, 166,
          244, 10, 242, 308, 28, 303, 230, 194, 268, 5, 63, 79, 257, 141, 56]
    c1 = [328, *range(345, 348), *range(358, 363), 310, 315, 11, 64, 81, 94, 108, 109, 179, 185, 213, 247, 317, 325,
          327, 52, 171, 173, 175, 188, 50, 51, 72, 104, 110, 111, 115, 118, 119, 124, 142, 143, 144, 147, 149, 165, 169,
          176, 192, 201, 239, 295, 321, 329, 334, *range(340, 345), 70, 102, 289, 291, 214, 215, 216, 224, 331, 19, 73,
          80, 96, 101, 178, 293, 36, 231, 306, 307, 309, 9, 55, 66, 74, 116, 4, 85, 261, 17, 65, 264, 41, 197, 297, 31,
          156, 157, 158, 159, 160, 161, 162, 163, 203, 205, 284, 333, 256, 290, 98, 100, 125, 126, 127, 128, 187, 222,
          236, 26, 37, 275, 313, 292, 3, 76, 88, 112, 258, 314, 332, 43, 326, 280, 97, 0, 91, 221, 193, 253, 18, 237,
          288, 301, 68, 190, 15, 223, 249, 62, 250, 285, 48, 245, 263, 84, 299, 302, 103, 20, 39, 278, 29, 60, 254, 251,
          23, 53, 255, 227, 44, 59, 57, 320, 167, 2, 71, 189, 7, 168, 238]
elif sh == 85:
    c0 = [*range(348, 352), 32, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46, 58, 259, 209, 220, 312, 1, 12, 30,
          61, 83, 89, 113, 145, 232, 235, 272, 281, 300, 311, 322, 34, 49, 202, 212, 217, 233, 269, 270, 271, 287, 305,
          330, 191, 226, 228, 273, 298, 6, 82, 92, 225, 319, 40, 47, 75, 77, 78, 117, 248, 267, 123, 208, 67, 219, 304,
          218, *range(335, 340), 13, 25, 35, 38, 86, 107, 121, 170, 172, 174, 204, 206, 207, 210, 211, 139, 294, 129,
          130, 131, 132, 133, 134, 135, 136, 137, 138, 90, 164, 177, 180, 181, 182, 183, 184, 195, 198, 199, 200, 21,
          33, 120, 122, 152, 153, 154, 155, 95, 252, 42, 240, 106, 323, 87, 246, 14, 186, 16, 274, 22]
    c1 = [328, *range(345, 348), *range(358, 363), 310, 315, 11, 64, 81, 94, 108, 109, 179, 185, 213, 247, 317, 325,
          327, 52, 171, 173, 175, 188, 50, 51, 72, 104, 110, 111, 115, 118, 119, 124, 142, 143, 144, 147, 149, 165, 169,
          176, 192, 201, 239, 295, 321, 329, 334, *range(340, 345), 70, 102, 289, 291, 214, 215, 216, 224, 331, 19, 73,
          80, 96, 101, 178, 293, 36, 231, 306, 307, 309, 9, 55, 66, 74, 116, 4, 85, 261, 17, 65, 264, 41, 197, 297, 31,
          156, 157, 158, 159, 160, 161, 162, 163, 203, 205, 284, 333, 256, 290, 98, 100, 125, 126, 127, 128, 187, 222,
          236, 26, 37, 275, 313, 292, 3, 76, 88, 112, 258, 314, 332, 43, 326, 280, 97, 0, 91, 221, 193]
elif sh == 171:
    c0 = [*range(348, 352), 32, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46, 58, 259, 209, 220, 312, 1, 12, 30,
          61, 83, 89, 113, 145, 232, 235, 272, 281, 300, 311, 322, 34, 49, 202, 212, 217, 233, 269, 270, 271, 287, 305,
          330, 191, 226, 228, 273, 298, 6, 82, 92, 225, 319, 40, 47, 75, 77, 78, 117, 248, 267, 123, 208, 67, 219, 304,
          218, *range(335, 340), 13, 25, 35, 38, 86, 107, 121, 170, 172, 174, 204, 206, 207, 210, 211, 139, 294]
    c1 = [328, *range(345, 348), *range(358, 363), 310, 315, 11, 64, 81, 94, 108, 109, 179, 185, 213, 247, 317, 325,
          327, 52, 171, 173, 175, 188, 50, 51, 72, 104, 110, 111, 115, 118, 119, 124, 142, 143, 144, 147, 149, 165, 169,
          176, 192, 201, 239, 295, 321, 329, 334, *range(340, 345), 70, 102, 289, 291, 214, 215, 216, 224, 331, 19, 73,
          80, 96, 101, 178, 293, 36, 231, 306, 307, 309, 9, 55, 66, 74, 116, 4, 85, 261, 17, 65, 264, 41, 197, 297, 31]
elif sh == 255:
    c0 = [*range(348, 352), 32, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46, 58, 259, 209, 220, 312, 1, 12, 30,
          61, 83, 89, 113, 145, 232, 235, 272, 281, 300, 311, 322, 34, 49, 202, 212, 217, 233, 269, 270, 271, 287, 305,
          330]
    c1 = [328, *range(345, 348), *range(358, 363), 310, 315, 11, 64, 81, 94, 108, 109, 179, 185, 213, 247, 317, 325,
          327, 52, 171, 173, 175, 188, 50, 51, 72, 104, 110, 111, 115, 118, 119, 124, 142, 143, 144, 147, 149, 165, 169,
          176, 192, 201]
elif sh == 341:
    c0 = []
    c1 = []
else:
    raise Exception('Number of shared features not implemented.')
c0.sort()
c1.sort()
print('client 0: {0}'.format(c0))
print('client 1: {0}'.format(c1))
shared = [x for x in range(0, 363) if x not in c0 and x not in c1]
print('shared: {0}'.format(shared))
fl = 'none'  # none, horizontal, or vertical
plus = True
adv_valid = True
rand_init = True
epochs = 100
inner = 100
fill = 0
test_size, valid_size = 0.2, 0.2
seed = 1226
model = LogisticRegression(max_iter=inner)
modelC = LogisticRegression(max_iter=inner)
head = 'IBMU4_Sh' + str(sh)
adv_opt = 'adam'
adv_beta = (0.9, 0.999)
adv_eps = 1e-8
alpha = 0.001
undersample = 4

adv = [*range(len(c0), len(c0)+len(shared))]
if fl.lower() != 'horizontal':
    c0.extend(shared)
if fl.lower() == 'vertical':
    c1.extend(shared)

np.random.seed(seed)
torch.manual_seed(seed)

# Load Data
filename = './data/IBM.csv'

# import dataset
df = pd.read_csv(filename, header=None)
df.drop(columns=df.columns[0], axis=1, inplace=True)

# undersample dominant class
ccol = df.columns[-1]
if undersample is not None:
    df_0 = df[~df[ccol]]
    df_1 = df[df[ccol]]
    df_0_under = df_0.sample(undersample * df_1.shape[0], random_state=seed)
    df = pd.concat([df_0_under, df_1], axis=0)
df[ccol] = df[ccol].replace({True: 1, False: 0})

# split classes & features
X = df.values[:, :-1]
y = df.values[:, -1]

# one-hot encode categorical variables
X = pd.DataFrame(X)
int_vals = np.array([X[col].apply(float.is_integer).all() for col in X.columns])
nunique = np.array(X.nunique())
cols = np.where(np.logical_and(np.logical_and(2 < nunique, nunique < 10), int_vals))[0]
X = pd.get_dummies(X, columns=X.columns[cols])  # convert objects to one-hot encoding
X = X.to_numpy()

X, X_test, y, y_test = train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=seed)
X, X_valid, y, y_valid = train_test_split(np.array(X), np.array(y), test_size=valid_size, random_state=seed)

advLogReg(X, X_valid, X_test, y, y_valid, y_test, fl, adv_valid, rand_init, epochs, inner, fill, adv_opt, adv_beta,
          adv_eps, alpha, c0, c1, shared, adv, model, head)


