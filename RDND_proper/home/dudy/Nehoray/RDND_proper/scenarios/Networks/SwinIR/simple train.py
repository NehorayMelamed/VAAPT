epoch = 0
epochs = 5000
epoch_losses = []
optimizer = optim.Adam(model.parameters(), lr=1e-5)
running_loss = 0.0
save_path = os.path.join(base_path,'new trained checkpoints')
predictions = []

### Concat Frames & Sigma: ###
B, T, C, H, W = network_input.shape
network_input = torch.cat((network_input, torch.zeros((B, T, 1, H, W)).to(network_input.device)), 2)

network_input = network_input, td  # For special vrt forward
return network_input, id, td, model

for a in train_dataloader:
    a = a
    break

x = a['output_frames_noisy']
y = a['output_frames_original'].cuda()

m = x[0][3]/255
n = y[0][3]/255

imshow_torch(m)
imshow_torch(n)


B, T, C, H, W = x.shape

x = torch.cat((x, torch.zeros((B, T, 1, H, W))),2).cuda()



### Get Batch: ###
def train_single_image():
    for epoch in range(epochs):
        print ('Starting epoch ', epoch+1)
        epoch_loss = 0.0
        optimizer.zero_grad()
        pred = model(x)
        diff = target - pred
        loss = diff.abs().mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print (loss.item(), 'epoch ', epoch+1)
        epoch_losses.append(epoch_loss)
        if epoch%100 == 0:
            save_path = os.path.join(base_path, 'new trained checkpoints/dvdnet_epoch_{}.pth'.format(epoch))
            torch.save(model.state_dict(), save_path)

def train():

    for epoch in range(epochs):
        print ('Starting epoch ', epoch+1)
        epoch_loss = 0.0
        for outputs_dict in train_dataloader:
            batch_output_frames = outputs_dict['output_frames_noisy']
            x = (batch_output_frames / (256 * 256 - 1)).cuda()
            target = (outputs_dict['center_frame_original'] / (256 * 256 - 1)).cuda()
            optimizer.zero_grad()
            pred = model(x)
            diff = target - pred
            loss = diff.abs().mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print (loss.item(), 'epoch ', epoch+1)
            epoch_losses.append(epoch_loss)
        if epoch%100 == 0:
            save_path = os.path.join(base_path, 'new trained checkpoints/dvdnet_epoch_{}.pth'.format(epoch))
            torch.save(model.state_dict(), save_path)

epoch = 0
epochs = 5000
epoch_losses = []
optimizer = optim.Adam(model.parameters(), lr=1e-4)
running_loss = 0.0
save_path = os.path.join(base_path,'new trained checkpoints')
predictions = []

for epoch in range(epochs):
    print ('Starting epoch ', epoch+1)
    epoch_loss = 0.0
    optimizer.zero_grad()
    pred = model((x,Train_dict))
    diff = y - pred
    loss = diff.abs().mean()
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    print (loss.item(), 'epoch ', epoch+1)
    epoch_losses.append(epoch_loss)
    # if epoch%100 == 0:
        # save_path = os.path.join(base_path, 'new trained checkpoints/dvdnet_epoch_{}.pth'.format(epoch))
        # torch.save(model.state_dict(), save_path)


imshow_torch(target[0],False,'original')
imshow_torch(pred,False, 'prediction in iter '+ str(epoch))
imshow_torch(x[0][2],False, 'downsample')
# #
plt.show(block=True)
#

train()

saved_epoch = 330
save_path = os.path.join(base_path,'new trained checkpoints/swin_epoch_{}.pth'.format(0))
# torch.save(model.state_dict(), save_path)

xd = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv').cuda()
xd.load_state_dict(torch.load(save_path))
# base_path = path_fix_path_for_linux('/media/mmm/DATADRIVE6/Omer')

for outputs_dict in train_dataloader:
    with torch.no_grad():

        batch_output_frames = outputs_dict['output_frames_noisy']
        x = (batch_output_frames / 255).cuda()
        pred =xd(x)
        target = (outputs_dict['center_frame_original'] / 255).cuda()
        imshow_torch(BW2RGB(RGB2BW(target)), False, 'original')
        imshow_torch(BW2RGB(RGB2BW(pred)), False, 'prediction in epoch ' + str(saved_epoch))
        imshow_torch(BW2RGB(RGB2BW((x))), False, 'downsample ' + str(saved_epoch))
        #
        plt.show(block=True)

plt.plot(range(epochs)[:epoch],epoch_losses)