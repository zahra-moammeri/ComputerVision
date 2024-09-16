@ -0,0 +1,24 @@
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
  def __init__(self):
    super(SRCNN, self).__init__()

    self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=(1,1), padding=(2,2))
    self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=(1,1), padding=(2,2))
    self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride=(1,1), padding=(2,2))


  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.conv3(x)

    return x
  

  """
  we are using the base architecture from the paper. 
  The only difference here is the extra padding to obtain the same size outputs as that of the input images.
  """