epoch = 0
epochs = 100
optimizer = optim.Adam(model.parameters(), lr=0.001)
running_loss = 0.0

target = outputs_dict['center_frame_original']/255
target = target.cuda()
x = x.cuda()
pred = model(x)
predictions = []

for epoch in range (iters):

    optimizer.zero_grad()
    pred  = model(x)
    diff = target - pred
    loss = diff.abs().mean()
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print (loss.item(), 'iter ' + str (iter))
    if iter%20 == 0:
        predictions.append(pred.clone())




imshow_torch(target,False,'original')
imshow_torch(pred,False, 'prediction in iter '+ str(iter))
imshow_torch(x,False, 'prediction in iter '+ str(iter))

plt.show(block=True)
