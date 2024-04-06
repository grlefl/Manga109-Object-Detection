import time
import numpy as np

def train_model(model, device, optimizer, train_loader, val_loader, epochs):
    # scheduler, save_dir, model_num, log_file

    # keep track of validation loss
    validation_loss = []
    best_loss = float('inf')

    start_time = time.time()  # start training

    for epoch in range(epochs):  # loop over the dataset multiple times

        # Training ----------------------------------------------------
        running_loss = 0.0
        for i, data in enumerate(train_loader):

            # THIS IS WEIRD
            images, targets = data[0].to(device), data[1]

            optimizer.zero_grad()                                   # zero the parameter gradients
            outputs = model(images)                                 # forward pass to get model predictions
            loss_dict = criterion(outputs, targets)                 # calculate loss
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()                                         # backward pass to compute gradients
            optimizer.step()                                        # update model parameters

            # add current bach loss to total epoch running loss
            running_loss += loss.item()
        epoch_loss = running_loss / (i + 1)
        print("Epoch: ", epoch, " train loss: ", '%.3f' % epoch_loss)

        # Validation --------------------------------------------------
        with torch.no_grad():
            running_loss = 0.0
            for i, data in enumerate(val_loader):

                # THIS IS WEIRD
                images, targets = data[0].to(device), data[1]

                outputs = model(images)                             # forward pass to get model predictions
                loss_dict = criterion(outputs, targets)             # calculate loss
                loss = sum(loss for loss in loss_dict.values())

                # add current bach loss to total epoch running loss
                running_loss += loss.item()
            epoch_loss = running_loss / (i + 1)

            # add each epoch loss to validation loss for early stopping
            validation_loss.append(epoch_loss)

            print("Epoch: ", epoch, " validation loss: ", '%.3f' % epoch_loss)

            # save the model if the validation loss is the best so far
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_weights = model.state_dict()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # check for early stopping based on patience
            if no_improvement_count >= patience:
                print("Stopping Early")
                break

    time_elap = (time.time() - start_time) // 60
    print('Finished Training in %d mins' % time_elap)