@ -0,0 +1,130 @@
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from torchvision.utils import save_image




plt.style.use('ggplot')
# demonstrates the "ggplot" style, which adjusts the style to emulate ggplot (a popular plotting package for R)


def psnr(label, outputs, max_val=1.):

    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    Note that the output and label pixels (when dealing with images) should
    be normalized as the `max_val` here is 1 and not 255.
    """

    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()

    """
    detach() method in PyTorch is used to separate a tensor from the           computational graph by returning a new tensor
    that doesn't require a gradient. If we want to move a tensor from the      Graphical Processing Unit (GPU) to the
    Central Processing Unit (CPU), then we can use detach() method

    You only need to call detach if the Tensor has associated gradients.       When detach is needed, you want to call
    detach before cpu. Otherwise, PyTorch will create the gradients            associated with the Tensor on the CPU
    then immediately destroy them when numpy is called.
    Calling detach first eliminates that superfluous step
    """

    diff = outputs - label
    rmse = math.sqrt(np.mean((diff) ** 2))
    if rmse == 0:
      return 100
    else:
      PSNR = 20 * math.log10(max_val/rmse)
      return PSNR
    


# save loss and psnr graphs

def save_plot(train_loss, val_loss, train_psnr, val_psnr ):

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/loss_100_200.png')
    plt.close()

    # PSNR plots
    plt.figure(figsize=(10,7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validation PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig('../outputs/psnr_100_200.png')
    plt.close()



# Save Model
def save_model_state(model):
# save the model to disk
    print('Saving model...')
    torch.save(model.state_dict(), '../outputs/model_100_200.pth')

"""
    In the above code block, the save_model_state function
    saves just the model state (trained weights) to the disk.
    We can use this for inference and share this as well because
    this will be smaller in size.
"""
    
def save_model(epochs, model, optimizer, criterion):

    #Function to save the trained model to disk.

    # Remove the last model checkpoint if present.
    torch.save({
                'epoch': epochs+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"../outputs/model_ckpt_100_200.pth")
                

"""
    save_model function, saves the entire model checkpoint
    to disk. This will save the state dictionary, the number
    of epochs trained for, the optimizer state dictionary, and
    the loss function as well. This will be a larger model because
    of the extra information but we can use this to resume training
    in the future if the case arises.

"""

    
    # Save Validation Images to Disk
    
"""
    the final helper function saves the reconstructed images from the validation 
    loop to the disk. This will give us a visual representation of how well 
    our model is learning. Again, we will not be saving these reconstructed images
    after every epoch. We will save them every 2 epochs as per the training script.
"""

def save_validation_results(outputs, epoch, batch_iter):


    # Function to save the Validation reconstructed images.


    save_image(
        outputs,
        f"../outputs/valid_results/val_sr_{epoch}_{batch_iter}.png"
    )