<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_sparseinst</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_sparseinst">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_sparseinst">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_sparseinst/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_sparseinst.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>


Run Sparseinst instance segmentation models.

![Cat instance segmentation](https://raw.githubusercontent.com/Ikomia-hub/infer_sparseinst/main/icons/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_sparseinst", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Inpect your result
display(algo.get_image_with_mask_and_graphics())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - default 'sparse_inst_r50_giam_aug': Name of the Sparseinst model. Additional models are available:
    - sparse_inst_r50vd_base
    - sparse_inst_r50_giam
    - sparse_inst_r50_giam_soft
    - sparse_inst_r50_giam_aug
    - sparse_inst_r50_dcn_giam_aug
    - sparse_inst_r50vd_giam_aug
    - sparse_inst_r50vd_dcn_giam_aug
    - sparse_inst_r101_giam
    - sparse_inst_r101_dcn_giam
    - sparse_inst_pvt_b1_giam
    - sparse_inst_pvt_b2_li_giam

- **conf_thres** (float) default '0.5': Confidence threshold for the prediction [0,1]
- **config_file** (str, *optional*): Path to the .yaml config file.
- **model_weight_file** (str, *optional*): Path to model weights file .pth. 


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_sparseinst", auto_connect=True)

algo.set_parameters({
    "model_name": "sparse_inst_r50_giam",
    "conf_thres": "0.5",
})

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Inpect your result
display(algo.get_image_with_mask_and_graphics())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_sparseinst", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

