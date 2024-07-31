import torch
from matplotlib import pyplot as plt


# save machine learning model to later with testing
def save_model(epoch_valid_loss, epoch, model, optimizer, output_dir):
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': epoch_valid_loss
    }, f"{output_dir}/last_model.pth")
    print('SAVING MODEL COMPLETE...')


# save validation and training loss plots
def save_loss_plot(valid_loss_list, train_loss_list, output_dir):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(valid_loss_list, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(f"{output_dir}/train_loss.png")
    figure_2.savefig(f"{output_dir}/valid_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')
