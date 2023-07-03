class Config:
    def __init__(self):
        # General arguments
        self.dataset = 'h36m'
        self.keypoints = 'cpn_ft_h36m_dbb'
        self.subjects_train = 'S1,S5,S6,S7,S8'
        self.subjects_test = 'S9,S11'
        self.subjects_unlabeled = ''
        self.actions = '*'
        self.checkpoint = 'checkpoint'
        self.checkpoint_frequency = 40
        self.resume = ''
        self.evaluate = ''
        self.render = False
        self.by_subject = False
        self.export_training_curves = False
        self.gpu = ['0',]
        self.local_rank = 0
        self.center_pose = 0

        # Model arguments
        self.stride = 1
        self.epochs = 200
        self.batch_size = 1024
        self.dropout = 0.
        self.learning_rate = 0.0001
        self.lr_decay = 0.99
        self.data_augmentation = True
        self.number_of_frames = 81
        self.number_of_kept_frames = 27
        self.number_of_kept_coeffs = 3
        self.depth = 4
        self.embed_dim_ratio = 32
        self.std = 0.0

        # Experimental
        self.subset = 1
        self.downsample = 1
        self.warmup = 1
        self.no_eval = False
        self.dense = False
        self.disable_optimizations = False
        self.linear_projection = False
        self.bone_length_term = True
        self.no_proj = False

        # Visualization
        self.viz_subject = None
        self.viz_action = None
        self.viz_camera = 0
        self.viz_video = None
        self.viz_skip = 0
        self.viz_output = None
        self.viz_export = None
        self.viz_bitrate = 3000
        self.viz_no_ground_truth = False
        self.viz_limit = -1
        self.viz_downsample = 1
        self.viz_size = 5

if __name__ == "__main__":
    # Usage
    config = Config()
    print(config.dataset)  # Access the parameters like this
