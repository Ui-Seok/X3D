# PySlowFast

# X3D

### install guide

contiribute facebookresearch/slowfast

[🚀️ click here](https://github.com/facebookresearch/slowfast)

### Before Start Check List

1. Prepare model
   * If use pre-trained model: Download X3D model(trained Kinetics-400)
   * [Download link](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md)
2. Edit .yaml file
   * TRAIN: false, TEST: false, DEMO: True
   * Edit {TEST.CHECKPOINT_FILE_PATH}
   * Can edit {DATA.SAMPLING_RATE}(now sampling_rate is 3)
   * Check {DEMO.LABEL_FILE_PATH}(now kinetics 400)
   * {DEMO.INPUT_VIDEO} and {DEMO.OUTPUT_FILE} can edit in code
3. Edit code
   * Can edit the data folder path in demo_net.py

### Start

```python
python tools/run_net.py --cfg {CONFIG_FILE_PATH}
```

## Contributors

PySlowFast is written and maintained by [Haoqi Fan](https://haoqifan.github.io/), [Yanghao Li](https://lyttonhao.github.io/), [Bo Xiong](https://www.cs.utexas.edu/~bxiong/), [Wan-Yen Lo](https://www.linkedin.com/in/wanyenlo/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/).

## Citing PySlowFast

If you find PySlowFast useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```
