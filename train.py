import time
import torch
from tqdm import tqdm
from save_results import save_loss_plot, save_model


# training loop
def train_model(device, model, optimizer, train_loader, valid_loader, num_epochs):
    output_dir = "./Results"

    patience = 10  # lower patience value?
    best_valid_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEPOCH {epoch + 1} of {num_epochs}")

        start = time.time()  # start timer and carry out training and validation

        train_loss_list = train(device, model, optimizer, train_loader)
        valid_loss_list = validate(device, model, valid_loader)

        epoch_train_loss = sum(train_loss_list) / len(train_loss_list)  # all train losses averaged
        epoch_valid_loss = sum(valid_loss_list) / len(valid_loss_list)  # all valid losses averaged

        end = time.time()
        print(f"Epoch #{epoch + 1}, Train Loss: {epoch_train_loss:.3f}, "
              f"Validation Loss: {epoch_valid_loss:.3f}, Duration: {((end - start) / 60):.3f} minutes")

        # early stopping implementation
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            patience_counter = 0
            save_model(best_valid_loss, epoch, model, optimizer, output_dir)  # save the best model
            save_loss_plot(valid_loss_list, train_loss_list, output_dir)      # save best loss plots
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Validation loss has not improved for {patience} epochs. Early stopping...")
                break

        # sleep for 5 seconds after each epoch
        time.sleep(5)


# run training iterations
def train(device, model, optimizer, train_loader):
    print('Training')
    train_loss_list = []

    # initialize tqdm progress bar
    prog_bar = tqdm(train_loader, total=len(train_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # various losses
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_list.append(loss_value)

        losses.backward()
        optimizer.step()

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


# run validation iterations
def validate(device, model, valid_loader):
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

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return valid_loss_list
