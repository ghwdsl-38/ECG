class MIT_BIH:
    def __init__(self):
        super(MIT_BIH, self).__init__()
        self.num_classes = 5
        self.class_names = ['N', 'S', 'V', 'F', 'Q']
        self.sequence_len = 186


class PTB:
    def __init__(self):
        super(PTB, self).__init__()
        self.num_classes = 2
        self.class_names = ['normal', 'abnormal']
        self.sequence_len = 188
