## placeholder class for batch elements

class Batch:
    """
    Placeholder for one batch
    """
    def __init__(self,
                 inp=None,
                 outp=None,
                 inp_lengths=None,
                 src_indexes=None):
        self.inp = inp
        self.inp_lengths = inp_lengths
        self.outp = outp
        self.src_indexes = src_indexes

    def to_device(self, device):
        self.inp = self.inp.to(device)
        self.outp = self.outp.to(device)
