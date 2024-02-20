# RoboKit
A toolkit for robotic tasks

## Features
- Zero-shot text-to-bbox approach for object detection using GroundingDINO.
- Zero-shot bbox-to-mask approach for object detection using SegmentAnything.
- Zero-shot classification using OpenAI CLIP.

## Getting Started

### Prerequisites
TODO
- Python 3.7 or higher (tested 3.9)
- torch (tested 2.0)
- torchvision

### Installation
```sh
git clone https://github.com/IRVLUTD/robokit.git && cd robokit 
pip install -r requirements.txt
python setup.py install
```

## Usage
- GroundingDINO + CLIP: [`test_gdino_clip.py`](test_gdino_clip.py)

## Roadmap

Future goals for this project include: 
TODO

## Acknowledgments

This project is based on the following repositories:
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [MobileSAM](git+https://github.com/ChaoningZhang/MobileSAM)
- [CLIP](https://github.com/openai/CLIP)

## License
This project is licensed under the MIT License