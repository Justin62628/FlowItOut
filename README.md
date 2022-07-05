# FlowItOut
Generate Flow from Trending Flow Algorithms

# Usage
1. Clone and Update Submodules(RAFT)
```bash
git clone https://github.com/Justin62628/FlowItOut
cd FlowItOut
git submodule update --init --recursive
```
2. Install Dependencies by `pip`
```bash
pip install requirements.txt
```
3. Download models from other places, put it in `models`
```bash
mkdir models
```
4. Prepare a video for test and Run
```bash
mkdir test
python flow_it_out.py -i test/test.mp4
```