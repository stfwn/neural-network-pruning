from copy import deepcopy
import torch.nn as nn

class ExpandedModule(nn.Module):
    def save_weights(self):
        print('Saving weights.')
        # Deepcopy to avoid just saving references
        self.saved_weights = deepcopy(list(self.parameters()))

    def reset_weights(self):
        with torch.no_grad():
            for saved, current in zip(self.saved_weights, self.parameters()):
                current.data = saved.data
