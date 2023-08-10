## UniversalNER 🚀

*UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition*<br/>
[Wenxuan Zhou*](https://wzhouad.github.io/), [Sheng Zhang*](https://sheng-z.github.io/), [Yu Gu](https://www.linkedin.com/in/aidengu/), [Muhao Chen](https://muhaochen.github.io/), [Hoifung Poon](https://www.microsoft.com/en-us/research/people/hoifung/) (*Equal Contribution)

[[Project Page](https://universal-ner.github.io/)] [[Demo](https://universal-ner.github.io/)] [[Paper](https://arxiv.org/abs/2308.03279)] [[Data](https://huggingface.co/Universal-NER)] [[Model](https://huggingface.co/Universal-NER)]

## Release
- **[8/10]** We have released the inference code for running the model checkpoints. The code for pretraining and evaluation will be released soon.

[![Model License](https://img.shields.io/badge/Model%20License-CC%20By%20NC%204.0-red.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
**Usage and License Notices**: The data, code, and model checkpoints are intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna, and ChatGPT.

## Install

1. Clone this repository and navigate to the folder
```bash
git clone https://github.com/universal-ner/universal-ner.git
cd universal-ner
```

2. Install the required packages
```Shell
pip install -r requirements.txt
```
## Demo

We use [vllm](https://github.com/vllm-project/vllm) for inference. The inference can be run with a single V100 16G GPU.

### Gradio Web UI

To launch a Gradio demo locally, run the following command:

```Shell
python -m src.serve.gradio_server \
    --model_path Universal-NER/UniNER-7B-type \
    --tensor_parallel_size 1 \
    --max_input_length 512
```

### CLI Inference

Run the following command:

```Shell
python -m src.serve.cli \
    --model_path Universal-NER/UniNER-7B-type \
    --tensor_parallel_size 1 \
    --max_input_length 512
```

## Citation

If you find UniversalNER helpful for your research and applications, please cite using this BibTeX:
```bibtex
@article{zhou2023universalner,
      title={UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition}, 
      author={Wenxuan Zhou and Sheng Zhang and Yu Gu and Muhao Chen and Hoifung Poon},
      year={2023},
      eprint={2308.03279},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
