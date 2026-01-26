
## Install

Set up environment:

```python
conda create -yn dobisvd python=3.10
conda activate dobisvd
pip install -e .
```

## Direct Use and Evaluation

We provide some decomposed models, please follow the instructions below to download and use them.

### Download Model

The model weight file is uploaded to Hugingface, you can download them by using the following command.

```python
git clone https://huggingface.co/Qinsi1/DobiSVD-Llama-2-7b-hf-0.4 ./results/compressed_model/Llama-2-7b-hf/DobiSVD-Llama-2-7b-hf-0.4
```

You can replace DobiSVD-Llama-2-7b-hf-0.4 with any release model and put them in the corresponding folder under ./results/compressed_model.

The current release model (0.4 means compression ratio = 40%):

| Model       | Remapping | 0.4                                                          | 0.6                                                          | 0.8                                                          |
| ----------- | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Llama-2-7b  | ✅         | [Qinsi1/DobiSVD-Llama-2-7b-hf-0.4](https://huggingface.co/Qinsi1/DobiSVD-Llama-2-7b-hf-0.4) | [Qinsi1/DobiSVD-Llama-2-7b-hf-0.6](https://huggingface.co/Qinsi1/DobiSVD-Llama-2-7b-hf-0.6) | [Qinsi1/DobiSVD-Llama-2-7b-hf-0.8](https://huggingface.co/Qinsi1/DobiSVD-Llama-2-7b-hf-0.8) |
| Llama-7b    | ✅         | [Qinsi1/DobiSVD-Llama-7b-hf-0.4](https://huggingface.co/Qinsi1/DobiSVD-Llama-7b-hf-0.4) | [Qinsi1/DobiSVD-Llama-7b-hf-0.6](https://huggingface.co/Qinsi1/DobiSVD-Llama-7b-hf-0.6) | [Qinsi1/DobiSVD-Llama-7b-hf-0.8](https://huggingface.co/Qinsi1/DobiSVD-Llama-7b-hf-0.8) |
| Llama-2-13b | ✅         | [Qinsi1/DobiSVD-Llama-2-13b-hf-0.4](https://huggingface.co/Qinsi1/DobiSVD-Llama-2-13b-hf-0.4) | [Qinsi1/DobiSVD-Llama-2-13b-hf-0.6](https://huggingface.co/Qinsi1/DobiSVD-Llama-2-13b-hf-0.6) | [Qinsi1/DobiSVD-Llama-2-13b-hf-0.8](https://huggingface.co/Qinsi1/DobiSVD-Llama-2-13b-hf-0.8) |
| Llama-13b   | ✅         | [Qinsi1/DobiSVD-Llama-13b-hf-0.4](https://huggingface.co/Qinsi1/DobiSVD-Llama-13b-hf-0.4) | [Qinsi1/DobiSVD-Llama-13b-hf-0.6](https://huggingface.co/Qinsi1/DobiSVD-Llama-13b-hf-0.6) | [Qinsi1/DobiSVD-Llama-13b-hf-0.8](https://huggingface.co/Qinsi1/DobiSVD-Llama-13b-hf-0.8) |
| Llama-2-7b  | ❌         | [Qinsi1/DobiSVD_Noremapping-Llama-2-7b-hf-0.4](https://huggingface.co/Qinsi1/DobiSVD-Llama-2-7b-hf-0.4) | [Qinsi1/DobiSVD_Noremapping-Llama-2-7b-hf-0.6](https://huggingface.co/Qinsi1/DobiSVD-Llama-2-7b-hf-0.6) | [Qinsi1/DobiSVD_Noremapping-Llama-2-7b-hf-0.8](https://huggingface.co/Qinsi1/DobiSVD-Llama-2-7b-hf-0.8) |

 

### Evaluate Model

You can quickly start evaluating the task performance of the downloaded model.

```python evaluate_speed_standalone.py --model_path <path_to_model>
```
