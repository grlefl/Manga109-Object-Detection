import time
import torch
from tqdm import tqdm


def train_model(device, model, optimizer, train_loader, train_epoch_hist, valid_loader, valid_epoch_hist, num_epochs):
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch + 1} of {num_epochs}")
        train_epoch_hist.reset()     # reset epoch training history
        valid_epoch_hist.reset()     # reset epoch validation history

        start = time.time()  # start timer and carry out training and validation

        train_loss = train(device, model, optimizer, train_loader, train_epoch_hist)
        val_loss = validate(device, model, valid_loader, valid_epoch_hist)

        end = time.time()
        print(f"Epoch #{epoch + 1}, Train Loss: {train_epoch_hist.value:.3f}, "
              f"Validation Loss: {valid_epoch_hist.value:.3f}, Duration: {((end - start) / 60):.3f} minutes")

        # early stopping
        if valid_epoch_hist.value < best_val_loss:
            best_val_loss = valid_epoch_hist.value
            patience_counter = 0
            # save_best_model(best_val_loss, epoch, model, optimizer)  # save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Validation loss has not improved for {patience} epochs. Early stopping...")
                break


        # save the best model till now if we have the least loss in the current epoch
        # save_best_model(
        #     val_loss_hist.value, epoch, model, optimizer
        # )
        # save_model(epoch, model, optimizer)             # save the current epoch model
        # save_loss_plot(OUT_DIR, train_loss, val_loss)   # save loss plot

        # sleep for 5 seconds after each epoch
        time.sleep(5)


# function for running training iterations
def train(device, model, optimizer, train_loader, train_epoch_hist):
    print('Training')
    train_loss_list = []

    # initialize tqdm progress bar
    prog_bar = tqdm(train_loader, total=len(train_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_list.append(loss_value)
        # train_epoch_hist.send(loss_value)

        losses.backward()
        optimizer.step()

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


# function for running validation iterations
def validate(device, model, valid_loader, val_loss_hist):
    print('Validating')
    valid_loss_list = []

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_loader, total=len(valid_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        valid_loss_list.append(loss_value)
        # val_loss_hist.send(loss_value)

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return valid_loss_list
