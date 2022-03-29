# horizon-ganns

Files for my MSc in Aerospace Engineering at ISAE-SUPAERO's Research Project titled "Using conditional GANs to build a human-robot interaction simulator"

## Dependencies
Dependencies install with conda:
```
conda create -n horizon
source activate horizon
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pandas scikit-learn -c conda-forge
```

## Folder structure

- **analysis** scripts to create plots/analyze results
- **datasets** classes for the processed datasets
- **launch_scripts** batch scripts to launch pando's training jobs
- **mnist** (c)gan testing implementations for mnist dataset
- **models** main models used in the S3 report
- **old_models** older files, not organized
- **preprocessing** scripts to process/filter/normalize the original dataset
- **utils** utility scripts
- **reports** RP reports


## Dataset

You can get the original dataset [from here](https://personnel.isae-supaero.fr/isae_ressources/caroline-chanel/horizon/). Place it in the main folder.

Run `preprocessing/data_preprocessing_normalize_observations.py`
To preprocess the dataset into the format specified in the S3 report, or just use the cached file (`processed_data/observations_cache`)

## Running the models
This will train and print statistics for the RCGAN model:
```bash
python3 -m models.rcgan
```
