# FlowItOut
Generate Optical Flow from trending Optical Flow Algorithms

# Usage
1. Clone and Update Submodules([RAFT](https://github.com/princeton-vl/RAFT))
```bash
git clone https://github.com/Justin62628/FlowItOut
cd FlowItOut
git submodule update --init --recursive
```
2. Install Dependencies by `pip`
```bash
pip install -r requirements.txt
```
2.1 Install `PyTorch` with CUDA, we are currently using CUDA 11.3
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
2.2 Put `ffmpeg` and `ffprobe` in your System Environment Parameters

For Windows users, put `ffmpeg.exe` and `ffprobe.exe` in the same folder for greatest convenience.
3. Download models from other places, put them in `models`
```bash
mkdir models
```
For RAFT, download `raft.pkl` at [Google Drive](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT)
4. Prepare a video for test and Run
```bash
mkdir test
python flow_it_out.py -i test/test.mp4
```
Typical Commands to use GMF:
```bash
python ./flow_it_out.py -i test/test.mp4 -s 960x540 --gmflow --model models/flownet.pkl
```
4.1 Please DO Remember to read notes by ```python flow_it_out.py --help```
```
usage: #### FlowItOut by Jeanna #### [-h] -i INPUT [-s RESIZE] [--model MODEL]
                                     [--raft | --gmflow | --others] [--small] 
                                     [--mixed_precision] [--alternate_corr]   

To generate flow by different trending algorithms

optional arguments:
  -h, --help            show this help message and exit

Basic Settings:
  -i INPUT, --input INPUT
                        Path of input video
  -s RESIZE, --resize RESIZE
                        Resized Resolution for flow, leave '0' for no-resize
  --model MODEL         restore checkpoint
  --raft
  --gmflow
  --others

RAFT Settings:
  Set the following parameters for RAFT

  --small               use small model
  --mixed_precision     use mixed precision
  --alternate_corr      use efficent correlation implementation, if
                        alternate_corr is not compiled, do not use

```
