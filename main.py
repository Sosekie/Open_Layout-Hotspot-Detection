from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.fftpack import dct
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

def dataload(image_path, save_path):
    image = Image.open(image_path)
    scale = 1200 / min(image.size)
    new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
    resized_image = image.resize(new_size, Image.ANTIALIAS)
    resized_image.save(save_path)
    return resized_image

def img_divide(resized_image, save_path = "./dataset/patches/1_"):
    square_width = resized_image.width // 12
    square_height = resized_image.height // 12
    squares = []
    for i in range(12):  # For each row
        for j in range(12):  # For each column
            left = j * square_width
            upper = i * square_height
            right = left + square_width
            lower = upper + square_height
            square = resized_image.crop((left, upper, right, lower))
            squares.append(square)
            square.save(save_path+str(i)+'_'+str(j)+'.jpg')
    return squares

def apply_dct(squares, select_lenth = 10):
    dct_squares = []
    for square in squares:
        square_grey = square.convert("L")
        square_array = np.array(square_grey)
        square_dct = dct(dct(square_array.T, norm='ortho').T, norm='ortho')
        square_dct = square_dct[0:select_lenth, 0:select_lenth]
        square_dct = (square_dct - np.min(square_dct)) / (np.max(square_dct) - np.min(square_dct))
        # square_dct = np.log(square_dct)
        dct_squares.append(square_dct)

    return dct_squares

def dct_visualize(dct_data, title='DCT Visualization', save_path="./dataset/dct_patches/1_dct.jpg"):

    # have a look at big ones：
    # square_dct_flattened = dct_data.flatten()
    # top_indices = np.argsort(square_dct_flattened)[-30:]
    # top_dct_values = square_dct_flattened[top_indices]
    # print("Top 10 DCT Coefficients: ", top_dct_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(dct_data.shape[0])
    y = np.arange(dct_data.shape[1])
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, dct_data, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_feature_map(dct_squares):
    all_flattened = np.concatenate([square.flatten() for square in dct_squares], axis=0)
    feature_map = all_flattened.reshape(-1, 12, 12)
    return feature_map

class Net(nn.Module):
    def __init__(self, in_channel):
        super(Net, self).__init__()
        # Assuming the input feature map size is 12x12x100 as from your previous message
        self.conv1_1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(3 * 3 * 32, 250)
        self.fc2 = nn.Linear(250, 2)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        
        x = x.view(-1, 3 * 3 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x

if __name__ == "__main__":
    image_path = "./dataset/1.jpg"
    save_path = "./dataset/1_resized.jpg"
    resized_image = dataload(image_path, save_path)
    squares = img_divide(resized_image)
    select_lenth = 4
    dct_squares = apply_dct(squares, select_lenth)
    # dct_visualize(dct_squares[68])
    feature_map = get_feature_map(dct_squares)

    # initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(in_channel = select_lenth*select_lenth).to(device)
    # get tensor
    feature_map = torch.tensor(feature_map, dtype=torch.float32).to(device)

    output = net(feature_map)
    print('True/False rate:', output)