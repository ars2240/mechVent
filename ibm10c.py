from fcmab import *
from fnn import *
from floaders import *
from itertools import chain

for sh in [1, 81, 171, 251, 341]:
    head = 'IBMU4_Sh' + str(sh)
    if sh == 1:
        c = [[141, 56, 227, 44, 59, 57, 320, 167, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 294, 297, 31, 74,
              116, 4, 85, 212, 217, 233, 269, 270, 271, 287, 305, 330],
             [2, 71, 189, 7, 168, 238, 146, 318, 256, 290, 98, 100, 125, 126, 127, 128, 129, 261, 17, 65, 264, 41, 197,
              67, 219, 144, 147, 149, 165, 169, 176, 192, 201, 202],
             [265, 148, 150, 151, 166, 244, 10, 242, 159, 160, 161, 162, 163, 203, 205, 284, 333, 304, 218,
              *range(335, 340), 13, 25, 35, 38, 86, 104, 110, 111, 115, 118, 119, 124, 142, 143],
             [308, 28, 303, 230, 194, 268, 5, 63, 120, 122, 152, 153, 154, 155, 156, 157, 158, 107, 121, 170, 172, 174,
              204, 206, 207, 281, 300, 311, 322, 34, 49, 50, 51, 72],
             [79, 257, 62, 250, 285, 48, 245, 263, 182, 183, 184, 195, 198, 199, 200, 21, 33, 210, 211, 214, 215, 216,
              224, 331, 19, 30, 61, 83, 89, 113, 145, 232, 235, 272],
             [84, 299, 302, 103, 20, 39, 278, 29, 326, 280, 97, 0, 90, 164, 177, 180, 181, 73, 80, 96, 101, 178, 293,
              36, 231, 171, 173, 175, 188, 209, 220, 312, 1, 12],
             [60, 254, 251, 23, 53, 255, 296, 282, 292, 3, 76, 88, 112, 258, 314, 332, 43, 306, 307, 309, 9, 55, 66,
              191, 226, 109, 179, 185, 213, 247, 317, 325, 327, 52],
             [8, 241, 277, 24, 27, 243, 234, 276, 14, 186, 187, 222, 236, 26, 37, 275, 313, 228, 273, 298, 6, 82, 92,
              225, 319, 58, 259, 310, 315, 11, 64, 81, 94, 108],
             [283, 105, 229, 316, 93, 140, 262, 260, 22, 95, 252, 42, 240, 106, 323, 87, 246, 40, 47, 75, 77, 78, 117,
              248, 267, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46],
             [279, 54, 253, 18, 237, 288, 301, 68, 190, 15, 223, 249, 91, 221, 193, 16, 274, 123, 208, 239, 295, 321,
              329, 334, *range(340, 345), 70, 102, 289, 291, 328, *range(345, 348), *range(358, 363), *range(348, 352),
              32]]
    elif sh == 81:
        c = [[130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 294, 297, 31, 74, 116, 4, 85, 212, 217, 233, 269, 270,
              271, 287, 305, 330],
             [256, 290, 98, 100, 125, 126, 127, 128, 129, 261, 17, 65, 264, 41, 197, 67, 219, 144, 147, 149, 165, 169,
              176, 192, 201, 202],
             [159, 160, 161, 162, 163, 203, 205, 284, 333, 304, 218, *range(335, 340), 13, 25, 35, 38, 86, 104, 110,
              111, 115, 118, 119, 124, 142, 143],
             [120, 122, 152, 153, 154, 155, 156, 157, 158, 107, 121, 170, 172, 174, 204, 206, 207, 281, 300, 311, 322,
              34, 49, 50, 51, 72],
             [182, 183, 184, 195, 198, 199, 200, 21, 33, 210, 211, 214, 215, 216, 224, 331, 19, 30, 61, 83, 89, 113,
              145, 232, 235, 272],
             [326, 280, 97, 0, 90, 164, 177, 180, 181, 73, 80, 96, 101, 178, 293, 36, 231, 171, 173, 175, 188, 209,
              220, 312, 1, 12],
             [292, 3, 76, 88, 112, 258, 314, 332, 43, 306, 307, 309, 9, 55, 66, 191, 226, 109, 179, 185, 213, 247, 317,
              325, 327, 52],
             [14, 186, 187, 222, 236, 26, 37, 275, 313, 228, 273, 298, 6, 82, 92, 225, 319, 58, 259, 310, 315, 11, 64,
              81, 94, 108],
             [22, 95, 252, 42, 240, 106, 323, 87, 246, 40, 47, 75, 77, 78, 117, 248, 267, 69, 99, 114, 196, 286, 324,
              *range(352, 358), 45, 46],
             [190, 15, 223, 249, 91, 221, 193, 16, 274, 123, 208, 239, 295, 321, 329, 334, *range(340, 345), 70, 102,
              289, 291, 328, *range(345, 348), *range(358, 363), *range(348, 352), 32]]
    elif sh == 171:
        c = [[139, 294, 297, 31, 74, 116, 4, 85, 212, 217, 233, 269, 270, 271, 287, 305, 330],
             [261, 17, 65, 264, 41, 197, 67, 219, 144, 147, 149, 165, 169, 176, 192, 201, 202],
             [304, 218, *range(335, 340), 13, 25, 35, 38, 86, 104, 110, 111, 115, 118, 119, 124, 142, 143],
             [107, 121, 170, 172, 174, 204, 206, 207, 281, 300, 311, 322, 34, 49, 50, 51, 72],
             [210, 211, 214, 215, 216, 224, 331, 19, 30, 61, 83, 89, 113, 145, 232, 235, 272],
             [73, 80, 96, 101, 178, 293, 36, 231, 171, 173, 175, 188, 209, 220, 312, 1, 12],
             [306, 307, 309, 9, 55, 66, 191, 226, 109, 179, 185, 213, 247, 317, 325, 327, 52],
             [228, 273, 298, 6, 82, 92, 225, 319, 58, 259, 310, 315, 11, 64, 81, 94, 108],
             [40, 47, 75, 77, 78, 117, 248, 267, 69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46],
             [123, 208, 239, 295, 321, 329, 334, *range(340, 345), 70, 102, 289, 291, 328, *range(345, 348),
              *range(358, 363), *range(348, 352), 32]]
    elif sh == 251:
        c = [[212, 217, 233, 269, 270, 271, 287, 305, 330], [144, 147, 149, 165, 169, 176, 192, 201, 202],
             [104, 110, 111, 115, 118, 119, 124, 142, 143], [281, 300, 311, 322, 34, 49, 50, 51, 72],
             [30, 61, 83, 89, 113, 145, 232, 235, 272], [171, 173, 175, 188, 209, 220, 312, 1, 12],
             [109, 179, 185, 213, 247, 317, 325, 327, 52], [58, 259, 310, 315, 11, 64, 81, 94, 108],
             [69, 99, 114, 196, 286, 324, *range(352, 358), 45, 46],
             [70, 102, 289, 291, 328, *range(345, 348), *range(358, 363), *range(348, 352), 32]]
    elif sh == 341:
        c = [[], [], [], [], [], [], [], [], [], []]
    else:
        raise Exception('Number of shared features not implemented.')
    shared = [x for x in range(0, 363) if x not in chain(*c)]
    adv = [*range(len(c[0]), len(c[0]) + len(shared))]
    for i in range(len(c)):
        c[i].sort()
        c[i].extend(shared)
    print(c)

    tr_loader, val_loader, te_loader = ibm_loader(batch_size=128, c=c, adv=adv, adv_valid=True, undersample=4)
    # tr_loader, val_loader, te_loader = adv_forest_loader(batch_size=128, adv_valid=True, c0=c0, c1=c1, head=head + '_best')
    model = FLNSH(feats=c, nc=10, classes=2)
    opt = torch.optim.Adam(model.parameters(), weight_decay=.01)
    loss = nn.CrossEntropyLoss()

    cmab = fcmab(model, loss, opt, nc=10, n=100, c='mabLin', head=head + '_FLNSH10c_Decay.01_RandPert', verbose=True)
    cmab.train(tr_loader, val_loader, te_loader)
