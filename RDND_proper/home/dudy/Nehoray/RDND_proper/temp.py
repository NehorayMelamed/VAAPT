import torch

def patchify_torch(input_tensor, patch_size, stride=None):
    if stride is None:
        stride = patch_size

    ### Get Shape: ###
    B,C,H,W = input_tensor.shape

    ### Patchify by adding doubling the amount of spatial dimensions (B,C,H,W) -> (B,C,H/P,W/P,P,P): ###
    input_tensor_patches = input_tensor.unfold(2, patch_size, stride).unfold(3, patch_size, stride)

    ### Stack N_patches on to Batch dimension (later on unpack it!!!) (B,C,H/P,W/P,P,P) -> (B*H*W/P^2,C,P,P): ###
    input_tensor_patches = input_tensor_patches.reshape(B, C, -1, patch_size, patch_size).reshape(-1, C ,patch_size, patch_size)

    return input_tensor_patches

x = torch.rand(6, 3, 192, 192).to(1)

patch_size = 16
stride = 1

x_patches_1 = patchify_torch(x,patch_size,1)
x_patches_2 = patchify_torch(x,patch_size,patch_size)
x_patches_1.shape
x_patches_2.shape

patches_arr = []
for i in range(16):
    patches_arr.append(patchify_torch(x,patch_size,i+1))

del x_patches_1
# torch.cuda.empty_cache()
patches_arr
