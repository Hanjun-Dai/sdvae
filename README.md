# sdvae
Syntax-Directed Variational Autoencoder for Structured Data (https://arxiv.org/abs/1802.08786)

#### 1. Download the data and pretrained model

Use the following dropbox link:

    https://www.dropbox.com/sh/621ufmvqgg5h2d8/AAARWPpuADNfPx8eu9E8y-rha?dl=0

Put everything under the 'dropbox' folder, or create a symbolic link with name 'dropbox':

    ln -s /path/to/your/downloaded/files dropbox
    
Finally the folder structure should look like this:

    sdvae (project root)
    |__  README.md
    |__  mol_vae
    |__  prog_vae
    |__  dropbox
    |__  |__ data
    |    |__ results
    |    |__ context_free_grammars
    |......
    
### 2. install dependencies and build c++ backend

The current code depends on pytorch 0.3.1. Most of the python dependencies can be installed by pip. 
However, the bayesian optimizaiton depends on a customized build of Theano. Please follow the 
instruction in GrammarVAE (https://github.com/mkusner/grammarVAE):

### below we will use mol_vae as the illustration for training/evaluation. The prog_vae works similarly.

### 3. Dataset pre-processing

Before training/evaluation, we need to cook the raw txt dataset. We use the mol_vae as illustration:

    cd mol_vae/data_processing
    ./run_data.sh
    ./run_cfg_dump.sh

The above two scripts will compile the txt data into binary file and cfg dump, correspondingly. 


### 4. Training

To train the model using GPU, run the following commands. You may also want to modify the parameters in 
the training script. 

    cd mol_vae/pytorch_train
    ./run_train.sh

The pretrained models are available under the dropbox folder, ``dropbox/results``. 

### 5. Evaluation

Before evaluation, we need to first dump the latent encodings of programs/molecules:

    cd mol_vae/pytorch_eval
    ./run_feature_dump.sh
    
To test the reconstruction, or sample from prior, please see the corresponding scripts under the same folder.

#### 5.1 Bayesian Optimization

To optimize the molecule property, run the bayesian optimization:

    cd mol_vae/mol_optimization
    ./run_bo.sh
    
After that, use the script ``get_final_results.py`` to collect the results. We use the same evaluation protocol
as in GrammarVAE(https://github.com/mkusner/grammarVAE). 

The results reported in the paper can be found under ``dropbox/results/zinc/bo``. If you use the same random seeds, 
then the exact same results should be expected.

#### 5.2 Sparse Gaussian Regression
   
To test the regression performance using the latent embeddings of molecules/programs:

    cd mol_vae/sparse_gp_regression
    ./run_regression.sh
    
Again, the 10 runs with different random seeds are reported, under ``dropbox/results/zinc/sgp``

#### 5.3 Visualization of Latent Space

To interpolate the latent space, do the following:

    cd mol_vae/visualize
    ./run_2dvis.sh

You may want to tune the gap, number of grids, etc., to see some reasonable visualization results.

### Reference

    @article{dai2018syntax,
      title={Syntax-Directed Variational Autoencoder for Structured Data},
      author={Dai, Hanjun and Tian, Yingtao and Dai, Bo and Skiena, Steven and Song, Le},
      journal={arXiv preprint arXiv:1802.08786},
      year={2018}
    }
