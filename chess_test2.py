# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import os
from matplotlib.patches import Rectangle
from torchvision.datasets import VisionDataset
from pycocotools import coco
import cv2
from copy import deepcopy
import albumentations as A
from albumentations.pytorch import ToTensorV2

# %%
def transform_image(split):
  if split == 'train':
    transform = A.Compose([
        A.Resize(1300, 2000),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomBrightness(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco'))
  else:
    transform = A.Compose([
        A.Resize(1300, 2000),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco'))
  return transform

# %%
class ChessDataset(VisionDataset):
  def __init__(self, root, split, transform=None, transforms=None, target_transform=None):
    super().__init__(root, transforms, transform, target_transform)
    self.root = root
    self.split = split
    self.coco = coco.COCO(os.path.join(self.root, self.split, '_annotations.coco.json'))
    self.ids = [value['id'] for value in self.coco.anns.values()]

    #some images contain more than one annotation so we are deleting ids where there is no anns
    self.ids = [id for id in self.ids if len(self.coco.loadAnns(self.coco.getAnnIds(id))) > 0]
    self.transforms = transforms

# %%
  def load_image(self, id):
    image_filename = self.coco.imgs[id]['file_name']
    self.image = cv2.imread(os.path.join(self.root, self.split, image_filename))
    return self.image

# %%
  def __len__(self):
    return len(self.ids)

# %%
  def __getitem__(self, index):
    id = self.ids[index]
    image = self.load_image(id)
    target = self.coco.loadAnns(self.coco.getAnnIds(id))
    target = deepcopy(target)
    bboxes = [t['bbox'] + [t['category_id']] for t in target]
    if self.transforms is not None:
      transformed = self.transforms(image=image, bboxes=bboxes)
    image = transformed['image']
    bboxes = transformed['bboxes']
    new_boxes = []
    for box in bboxes:
      x_min = box[0]
      y_min = box[1]
      x_max = x_min + box[2]
      y_max = y_min + box[3]
      new_boxes.append([x_min, y_min, x_max, y_max])
    final_target = {}
    final_target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)
    final_target['labels'] = torch.tensor([value['category_id'] for value in self.coco.loadAnns(self.coco.getAnnIds(id))], dtype=torch.int64)
    # final_target['image_id'] = torch.tensor([value['image_id'] for value in train_coco.loadAnns(train_coco.getAnnIds(id))], dtype=torch.int64)
    # final_target['area'] = torch.tensor([value['area'] for value in train_coco.loadAnns(train_coco.getAnnIds(id))], dtype=torch.int64)
    # final_target['iscrowd'] = torch.tensor([value['iscrowd'] for value in train_coco.loadAnns(train_coco.getAnnIds(id))], dtype=torch.int64)
    return image.div(255.), final_target

# %%
class ChessBoard:
  def __init__(self):
      self.height = 0
      self.width = 0
      self.h_spacing = 0
      self.w_spacing = 0
      self.real_board = np.zeros((8, 8, 2))
      self.sim_board = np.zeros((8, 8, 2))
      for j in range(8):
          for i in range(8):
              self.sim_board[j, i] = [0 + 1*i, 0 + 1*j]
  def create_real_board(self, w_rook1, w_rook2, b_rook1, b_rook2):
      w_rook1 = [w_rook1[0] + (w_rook1[2] - w_rook1[0]) / 2, w_rook1[3] - 10]
      w_rook2 = [w_rook2[0] + (w_rook2[2] - w_rook2[0]) / 2, w_rook2[3] - 10]
      b_rook1 = [b_rook1[0] + (b_rook1[2] - b_rook1[0]) / 2, b_rook1[3] - 10]
      b_rook2 = [b_rook2[0] + (b_rook2[2] - b_rook2[0]) / 2, b_rook2[3] - 10]
      if w_rook1[1] < w_rook2[1]:
        self.w_upper_rook = w_rook1
        self.w_bottom_rook = w_rook2
      else:
        self.w_upper_rook = w_rook2
        self.w_bottom_rook = w_rook1
      if b_rook1[1] < b_rook2[1]:
        self.b_upper_rook = b_rook1
        self.b_bottom_rook = b_rook2
      else:
        self.b_upper_rook = b_rook2
        self.b_bottom_rook = b_rook1
      self.height = np.abs(w_rook1[1] - w_rook2[1])
      self.upper_width = np.abs(self.b_upper_rook[0] - self.w_upper_rook[0])
      self.bottom_width = np.abs(self.w_bottom_rook[0] - self.b_bottom_rook[0])


      self.width_differenece = (self.bottom_width - self.upper_width) / 2
      self.width_increment =  self.width_differenece / 7
      
      self.w_spacing = self.upper_width / 7
      self.h_spacing = self.height / 7
      starting_point = self.w_upper_rook

      # set starting offsets for width and height
      self.w_offset = (self.w_bottom_rook[0] - self.w_upper_rook[0]) / 7
      self.h_offset = (self.w_upper_rook[1] - self.b_upper_rook[1]) / 7

      self.heights = self._calculate_heights(self.upper_width, self.bottom_width, self.height, self.w_bottom_rook[1], self.w_upper_rook[1] + 10)
            
      # create board
      for j in range(8):
          for i in range(8):
            #   self.real_board[j, i] = [starting_point[0] - self.width_increment*j + ((self.w_spacing + 2*self.width_increment*j/7))*i, starting_point[1] + self.h_spacing*j]
              self.real_board[j, i] = [starting_point[0] - self.width_increment*j + ((self.w_spacing + 2*self.width_increment*j/7))*i, self.heights[j]]
  
  def _calculate_h_and_width(self, bottom_width, upper_width, height):
    new_height = bottom_width*height / (bottom_width + upper_width)
    ratio = new_height / height
    new_width = bottom_width - (bottom_width - upper_width)*ratio
    return new_height, new_width
  
  def _calculate_heights(self, upper_width, bottom_width, height, bottom_height, upper_height):
    heights = []
    a = bottom_width
    b = upper_width
    H = height + height/14

    # heights.append(upper_height)

    half_h, half_width = self._calculate_h_and_width(a, b, H)
    heights.append(bottom_height - half_h)
    
    bottom_quarter_h, bottom_quarter_width = self._calculate_h_and_width(a, half_width, half_h)
    heights.append(bottom_height - bottom_quarter_h)

    upper_quarter_h, upper_quarter_width = self._calculate_h_and_width(half_width, b, (H - half_h))
    heights.append(bottom_height - half_h - upper_quarter_h)
    
    bottom_bottom_h, bottom_bottom_width = self._calculate_h_and_width(a, bottom_quarter_width, bottom_quarter_h)
    heights.append(bottom_height - bottom_bottom_h)

    bottom_upper_h, bottom_upper_width = self._calculate_h_and_width(bottom_quarter_width, half_width, (half_h - bottom_quarter_h))
    heights.append(bottom_height - bottom_quarter_h - bottom_upper_h)

    upper_upper_h, upper_upper_width = self._calculate_h_and_width(upper_quarter_width, b, (H - half_h - upper_quarter_h))
    heights.append(bottom_height - half_h - upper_quarter_h - upper_upper_h)

    upper_bottom_h, upper_bottom_width = self._calculate_h_and_width(half_width, upper_quarter_width, upper_quarter_h)
    heights.append(bottom_height - half_h - upper_bottom_h)

    heights.append(bottom_height)

    return sorted(heights)

  def draw_board(self, ax):
    # fig, ax = plt.subplots()
    color_id = 1
    for j in range(8):
      color_id += 1
      for i in range(8):
          if color_id % 2 == 0:
            color = 'black'
          else:
              color = 'white'
          color_id += 1
          ax.add_patch(Rectangle((self.sim_board[j, i][0], -self.sim_board[j, i][1]), 1, 1,edgecolor = 'black', facecolor = color))
      ax.plot()

# %%
prediction_bound = 0.7

# %%
test_data = ChessDataset('/home/jakub/Downloads/chess_coco', 'test', transforms=transform_image('test'))

# %%
# first image for calibration
img, _ = test_data[11]
img_int = torch.tensor(img*255, dtype=torch.uint8)

# %%
model = torch.load(f'/home/jakub/torch_models/chess/model', map_location=torch.device('cpu'))
model.eval()
# %%
prediction = model([img])
categories = [cat['name'] for cat in test_data.coco.cats.values()]
# %%
rectangle_bboxes = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in prediction[0]['boxes'][prediction[0]['scores'] > prediction_bound]]
# %%
categories
# %%
# detach() is used to avoid errors realted to tensor requiring gradient
white_rooks = [bbox.detach().numpy() for bbox in prediction[0]['boxes'][prediction[0]['labels'] == 13]]
black_rooks = [bbox.detach().numpy() for bbox in prediction[0]['boxes'][prediction[0]['labels'] == 7]]
# %%
detections = [bbox.detach().numpy() for bbox in prediction[0]['boxes'][prediction[0]['scores'] > prediction_bound]]
labels = [categories[label] for label in prediction[0]['labels'][prediction[0]['scores'] > prediction_bound]]
# %%
fig, ax = plt.subplots(2, 1)
ax[0].imshow(img_int.permute(1, 2, 0))
for i, box in enumerate(rectangle_bboxes):
    ax[0].add_patch(Rectangle((box[0].item(), box[1].item()), box[2].item(), box[3].item(), fill=False))
    # ax[0].text(box[0].item(), box[1].item(), categories[prediction[0]['labels'][prediction[0]['scores'] > prediction_bound][i].item()], fontsize='xx-small', color='red')
    ax[0].text(box[0].item(), box[1].item(), labels[i], fontsize='xx-small', color='red')

# %%
chessboard = ChessBoard()
chessboard.create_real_board(white_rooks[0], white_rooks[1], black_rooks[0], black_rooks[1])
ax[0].scatter(chessboard.real_board[:,:,0].flatten(), chessboard.real_board[:,:,1].flatten())
# show full screen
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()

# %%
indexes = np.ones((len(detections), 2))
for k in range(len(labels)):
    min = 10000
    for j in range(8):
        for i in range(8):
            figure = [detections[k][2] - (detections[k][2] - detections[k][0]) / 2, detections[k][3]]
            distance = math.sqrt((chessboard.real_board[j, i][0] - figure[0])**2 + (chessboard.real_board[j, i][1] - figure[1])**2)
            if distance < min:
                min = distance
                indexes[k] = [j, i]

# %%
# itarating over test images
for i in range(15):
    img, _ = test_data[i]
    img_int = torch.tensor(img*255, dtype=torch.uint8)
    prediction = model([img])

    rectangle_bboxes = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in prediction[0]['boxes'][prediction[0]['scores'] > prediction_bound]]
    detections = [bbox.detach().numpy() for bbox in prediction[0]['boxes'][prediction[0]['scores'] > prediction_bound]]
    labels = [categories[label] for label in prediction[0]['labels'][prediction[0]['scores'] > prediction_bound]]
    
    # draw image with boxes
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_int.permute(1, 2, 0))
    
    # scatter 'real_board' points
    # ax[0].scatter(chessboard.real_board[:,:,0].flatten(), chessboard.real_board[:,:,1].flatten())

    # draw bounding boxes on real image with labels
    # for i, box in enumerate(rectangle_bboxes):
    #     ax[0].add_patch(Rectangle((box[0].item(), box[1].item()), box[2].item(), box[3].item(), fill=False))
    #     ax[0].text(box[0].item(), box[1].item(), categories[prediction[0]['labels'][prediction[0]['scores'] > prediction_bound][i].item()], fontsize='xx-small', color='red')

    # draw simulated board
    indexes = np.zeros((len(detections), 2))
    points = []
    for k in range(len(labels)):
      min = 10000
      for j in range(8):
          for i in range(8):
              figure = [detections[k][0] + (detections[k][2] - detections[k][0]) / 2, detections[k][3] - 10]
              distance = math.sqrt((chessboard.real_board[j, i][0] - figure[0])**2 + (chessboard.real_board[j, i][1] - figure[1])**2)
              if distance < min:
                  min = distance
                  indexes[k] = [j, i]
                  point = [detections[k][0] + (detections[k][2] - detections[k][0]) / 2, detections[k][3] - 10]

      points.append(point)
    points = np.array(points)

    # scatter detected points for distance comparison 
    # if len(points) > 0:
    #   ax[0].scatter(points[:, 0], points[:, 1])

    # plotting 2D board
    chessboard.draw_board(ax[1])

    for i, index in enumerate(indexes):
        j, k = index
        j = int(j)
        k = int(k)
        ax[1].text(int(chessboard.sim_board[j, k, 0]), -int(chessboard.sim_board[j, k, 1]) + 0.5, labels[i], fontsize='xx-small', color='red')
    ax[1].set_xlim((0, 10))
    ax[1].set_ylim((-10, 1))
    ax[0].axis('off')
    ax[1].axis('off')

    # show maximized
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()

# %% [markdown]
# %%<br>
# detach() is used to avoid errors realted to tensor requiring gradient




