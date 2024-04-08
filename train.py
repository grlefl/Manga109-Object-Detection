# start the training epochs
def train(num_epochs,train_loss_hist,val_loss_hist,time,train_loader,val_loader,model,optimizer):
    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch + 1} of {num_epochs}")
        train_loss_hist.reset()     # reset epoch training history
        val_loss_hist.reset()       # reset epoch validation history

        start = time.time()  # start timer and carry out training and validation

        train_loss = train(train_loader, model)
        val_loss = validate(val_loader, model)
        print(f"Epoch #{epoch + 1} train loss: {train_loss_hist.value:.3f}, validation loss: {val_loss_hist.value:.3f}")

        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        # save the best model till now if we have the least loss in the current epoch
        # save_best_model(
        #     val_loss_hist.value, epoch, model, optimizer
        # )
        # save_model(epoch, model, optimizer)             # save the current epoch model
        # save_loss_plot(OUT_DIR, train_loss, val_loss)   # save loss plot

        # sleep for 5 seconds after each epoch
        time.sleep(5)


# function for running training iterations
def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


# function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


