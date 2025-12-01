BATCH_SIZE = 16
EPOCH_NUMBER = 200
lr = 2e-4
image_size = 256
class_dict_path = './model_utils/class_dict.csv'

dataset = "BUSI"
TRAIN_IMG_ROOT = "./dataset/images"
TRAIN_LBL_ROOT = "./dataset/masks"

VAL_IMG_ROOT = './dataset/training/images'
VAL_LBL_ROOT = './dataset/training/mask'

# 测试集s
TEST_IMG_ROOT = './dataset/test/images'
TEST_LBL_ROOT = './dataset/test/mask'
