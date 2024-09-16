import os

from torch import Tensor
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

# import tqdm

# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 이미지 디렉토리 경로
dir_path = '/home/dolggul2/data/bracket_3000/'

# 이미지 목록 가져오기
file_list = [file for file in os.listdir(dir_path) if file.endswith('.jpg')]

# 이미지,레이블을 X,Y 리스트에 담기
def read_img(src: str, file: str) -> Tensor:
    img_path: str = os.path.join(src, file)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # torchvision.io.imread 대신 이걸 쓸 상황이 있을까?
    img: Tensor = read_image(img_path)
    return img

X, y = [], []
for file in file_list:
    X.append(read_img(dir_path, file))
    y.append(file[:-4])

# X,Y 리스트로 train, test 데이터셋으로 나누기

# feature에 대한 transform인데, 뜻이 명확하지 않은것같아.
# 입력값의 형태를 변환시키고, 값을 변환시키고 싶었어.
custom_transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float), # convert value to range 0 ~ 1
    transforms.Lambda(lambda x: x.view(-1)) # specify shape like ([4, 2, -1])
])

def label_to_tensor(label):
    return torch.tensor(float(label))


class CustomDataset(Dataset):
    def __init__(self, dir_path: str, file_list: list[str], transform=None, target_transform=None) -> None:
        self.file_list = file_list
        self.dir_path = dir_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx) -> tuple[Tensor, torch.float]: # label이 str이면 dataloader에서 tuple로 반환
        img: Tensor = read_img(self.dir_path, self.file_list[idx])
        label = self.file_list[idx][:-4]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label



dataset = CustomDataset(dir_path, file_list, custom_transform, label_to_tensor)


# dataset을 분류하고 싶어.
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])


# dataset별로 feature, label로 분류하고 싶어.
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
# train_features, train_labels = next(iter(train_dataloader)) # Tensor, Tuple??
# print(type(train_labels))

class AutoEncoderModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,20),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(20, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,3136),
            torch.nn.Sigmoid(), # TODO: 여기서 왜 Sigmoid를 쓰지?
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

# 하이퍼파라미터는 딕셔너리도 담아서 전달해주는게 어떨까?
ae = AutoEncoderModel()
ae.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
epochs = 100

def train_loop(dataloader, model, criterion, optimizer, epochs):
    model.to(device) # 모델은 inplace 연산
    model.train() # 이건 뭘 하는거지?

    for epoch in range(epochs):
        for X, y in tqdm(train_dataloader):
            X = X.to(device) # 데이터는 to가 inplace 연산이 아님, 직접 다시 할당해줘
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred, X) #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




train_loop(train_dataloader, ae, criterion, optimizer, epochs)


# 테스트 루프 및 결과 시각화
def test_and_visualize(dataloader, model, num_images_to_show=5):
    model.eval()  # 평가 모드 설정

    original_images = []
    generated_images = []

    with torch.no_grad():  # 평가 시에는 그래디언트 계산 비활성화
        for X, _ in dataloader:
            X = X.to(device)
            pred = model(X)

            # 원본 이미지와 생성된 이미지 저장
            for i in range(num_images_to_show):
                original_img = X[i].view(56, 56).cpu().numpy()  # 56x56 크기로 복원
                generated_img = pred[i].view(56, 56).cpu().numpy()

                original_images.append(original_img)
                generated_images.append(generated_img)

            # 원하는 이미지 수만큼 비교 후 종료
            if len(original_images) >= num_images_to_show:
                break

    # 시각화: 한 그림에 원본 5개, 생성된 이미지 5개 비교
    fig, axes = plt.subplots(2, num_images_to_show, figsize=(15, 5))

    for i in range(num_images_to_show):
        # 첫 번째 행: 원본 이미지
        axes[0, i].imshow(original_images[i], cmap='gray')
        axes[0, i].axis('off')  # 축 제거
        axes[0, i].set_title(f"Original {i + 1}")

        # 두 번째 행: 생성된 이미지
        axes[1, i].imshow(generated_images[i], cmap='gray')
        axes[1, i].axis('off')  # 축 제거
        axes[1, i].set_title(f"Generated {i + 1}")

    plt.tight_layout()
    plt.show()


# 테스트 데이터셋으로 결과 확인
test_and_visualize(test_dataloader, ae, num_images_to_show=5)

# image = read_img(dir_path, file_list[0])
# print(image.shape)
