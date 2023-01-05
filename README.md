
## Setup

Download the data from: https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg and put it under `fairness_last_layer/celeba`. The folder should have the following content:
```
fairness_last_layer/celeba/
├── img_align_celeba
│   ├── 000001.jpg
│   ├── 000002.jpg
│   ├── 000003.jpg
│   └── ...
├── identity_CelebA.txt
├── list_bbox_celeba.txt
├── list_landmarks_celeba.txt
├── list_eval_partition.txt
├── list_attr_celeba.txt
├── list_landmarks_align_celeba.txt

```

The attribute list of CelebA data is as follows:
![List-of-the-40-face-attributes-provided-with-the-CelebA-database](https://user-images.githubusercontent.com/57878927/210681993-83dbfceb-1b80-438e-938b-0eba4d727376.png)


## Run

```
cd fairness_last_layer
pip install -r requirements.txt
python CelebA.py  --method M2  --alpha 5  --constraint EO  --seed 2023
````
- constraint: `EO`, `AE`, `DI`, `MMF`


