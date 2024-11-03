import streamlit as st
import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import face_recognition
from torch import nn
from torchvision import models
from torch.utils.data.dataset import Dataset
import os

# Set page config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🎥",
    layout="centered"
)

class Model(nn.Module):
    """
    A neural network model for deepfake detection.
    
    Args:
        num_classes (int): Number of output classes.
        latent_dim (int): Dimension of the latent space.
        lstm_layers (int): Number of LSTM layers.
        hidden_dim (int): Dimension of the hidden state in LSTM.
        bidirectional (bool): If True, use bidirectional LSTM.
    """
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            tuple: Feature map and logits.
        """
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

class ValidationDataset(Dataset):
    """
    A dataset class for validation videos.
    
    Args:
        video_names (list): List of video file paths.
        sequence_length (int): Number of frames to extract from each video.
        transform (callable, optional): A function/transform to apply to the frames.
    """
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        """
        Returns number of videos in dataset.
        
        Returns:
            int: Number of videos.
        """
        return len(self.video_names)

    def __getitem__(self, idx):
        """
        Returns a sequence of frames from a video.
        
        Args:
            idx (int): Index of the video.
        
        Returns:
            torch.Tensor: A tensor containing the frames.
        """
        video_path = self.video_names[idx]
        frames = []
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        """
        Extracts frames from a video file.
        
        Args:
            path (str): Path to the video file.
        
        Yields:
            numpy.ndarray: Extracted frame.
        """
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def predict(model, img):
    """
    Predicts whether a video is real or fake.
    
    Args:
        model (Model): The deepfake detection model.
        img (torch.Tensor): Input tensor containing video frames.
    
    Returns:
        list: Prediction and confidence score.
    """
    sm = nn.Softmax()
    fmap, logits = model(img.to())
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return [int(prediction.item()), confidence]

def detect_fake_video(video_path):
    """
    Detects whether a video is real or fake.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        list: Prediction and confidence score.
    """
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    video_dataset = ValidationDataset([video_path], sequence_length=20, transform=train_transforms)
    model = Model(2)
    
    path_to_model = 'df_model.pt'
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    
    prediction = predict(model, video_dataset[0])
    return prediction

def main():
    st.title("Deepfake Detection System")
    st.write("Upload a video to check if it's real or fake")

    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        # save uploaded file temporarily
        temp_path = "temp_video." + uploaded_file.name.split('.')[-1]
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner('Analyzing video...'):
            try:
                prediction = detect_fake_video(temp_path)
                
                result = "REAL" if prediction[0] == 1 else "FAKE"
                confidence = prediction[1]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Result", result)
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}%")

                if result == "FAKE":
                    st.warning("This video appears to be manipulated!", icon="⚠️")
                else:
                    st.success("This video appears to be authentic!", icon="✅")

            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
            finally:
                # clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main()