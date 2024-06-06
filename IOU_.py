#     tn=0
#     fp=0
#     fn=0
#     for i in range(128):
#         for j in range(128):
#             if binary_mask[ i, j, 0] == 1 and mask_image[0, i, j, 0] == 1:
#                 tp += 1
#             elif binary_mask[ i, j, 0] == 0 and mask_image[0, i, j, 0] == 0:
#                 tn += 1
#             elif binary_mask[ i, j, 0] == 1 and mask_image[0, i, j, 0] == 0:
#                 fp += 1
#             elif binary_mask[ i, j, 0] == 0 and mask_image[0, i, j, 0] == 1:
#                 fn += 1
#     if tp+fp+fn==0:
#         list1.append(1)
#     else:
#         list1.append(tp/(tp+fp+fn))
#     #list1.append([tp,tn,fp,fn])
