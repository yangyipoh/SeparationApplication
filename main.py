"""Application front end for the model"""
import io
from typing import Optional, Tuple
import yaml
import streamlit as st
import torch
from torch import Tensor
import torchaudio
import torchaudio.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.masknet import MaskNet, MaskNetConfig


@st.cache_resource
def load_model():
    """Load model with caching"""
    # model params
    model_path = 'models/masknet/model_best.pt'
    model_config = 'models/masknet/model.yaml'
    device = torch.device('cpu')

    with open(model_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model = MaskNet(MaskNetConfig(**config))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    return model


def plotter(t, data, xlabel='time (s)', ylabel='amplitude', title=''):
    """plot data"""
    fig, ax = plt.subplots()
    ax.plot(t, data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)


@torch.no_grad()
def parse_separate(data:Tensor, model:MaskNet) -> Tuple[Tensor, Tensor]:
    """separate data into heart and lung"""
    data = torch.unsqueeze(data, dim=0)     # pylint: disable=no-member
    output = model(data)
    heart = output[0, 0, :]
    lung = output[0, 1, :]
    return heart, lung


def parse_input() -> Optional[Tensor]:
    """read uploaded file and return PyTorch Tensor"""
    st.header('Input Audio Files')
    uploaded_file = st.file_uploader("Choose an audio file (only supports wav)", type=['wav'])

    if uploaded_file is None:
        return None
    audio_byte = uploaded_file.read()
    data, fs = torchaudio.load(io.BytesIO(audio_byte), backend="ffmpeg")  # pylint: disable=no-member
    data = data[0]      # only select one channel
    new_fs = 4_000
    data = F.resample(data, fs, new_fs)

    parse_audio(data.numpy(), new_fs, 'Input waveform')
    return data


def parse_output(heart:np.ndarray, lung:np.ndarray):
    """parse output based on heart and lung sounds"""
    fs = 4_000
    st.header('Output Audio Files')
    st.subheader('Heart Sounds')
    parse_audio(heart, fs, title='Heart waveform')

    st.subheader('Lung Sounds')
    parse_audio(lung, fs, 'Lung waveform')


def parse_audio(audio:np.ndarray, fs:int=4_000, title:str='') -> None:
    """parse audio type content"""
    t = np.arange(len(audio))/fs
    plotter(t, audio, title=title)
    st.audio(audio, sample_rate=fs)


def sidebar():
    """sidebar information"""
    gh_url = r'https://github.com/yangyipoh/SeparationApplication'
    dcker_url = r'https://hub.docker.com/r/yangyipoh/sep_app'
    st.caption("Created by Yang Yi Poh: Yang.Poh@monash.edu")
    st.divider()
    st.subheader('Application for the paper "Neonatal Chest Sound Separation using Deep Learning"')
    st.write(f'The following application can be found on [Github]({gh_url}) and [Docker]({dcker_url}) for deployment')

def main():
    """main GUI function"""
    # sidebar info
    with st.sidebar:
        sidebar()

    st.title('Neonatal Chest Sound Separation')
    model = load_model()

    data = parse_input()

    start_separate = st.button('Separate!', type="primary")

    if start_separate and data is not None:
        heart, lung = parse_separate(torch.unsqueeze(data, dim=0), model)  # pylint: disable=no-member
        parse_output(heart.numpy(), lung.numpy())


if __name__ == '__main__':
    main()
