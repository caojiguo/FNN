## Data and R Codes used for each application and simulation studies presented at the manuscript entitled "Deep Learning with Functional Inputs"

### This markdown file has information on each script for the paper "Deep Learning with Functional Inputs". It also has information on what package versions are required to make sure that they all run.

---------------------------------------------------------------

- FNN.R = The main function file containing the backbone for every script. It contains the core functions for FNNs and also has great overlap with the FuncNN package. We use these functions as opposed to the associated package because the package was developed after this paper was initially submitted.

- FNN_Figure2.R = This script produces Figure 2 in the manuscript.

- FNN_AblatioStudies.R = This script produces the Ablation results found in the supplementary document.

- FNN_BikeExample.R = This script contains all the main results for the bike example in the manuscript (Found in Section 3.2).

- FNN_TecatorExample.R = This script contains all the results for the tecator example in the supplementary document.

- FNN_TecatorNeuralNetworks.R = This script contains all the results for the tecator example in the manuscript (Found in Section 3.1).

- FNN_WeatherExample.R = This script contains all the main results for the weather example in the manuscript (Found in Section 3.3).

- FNN_FunctionalWeights.R = This script produces the Functional Weight examples seen in the Bike and Weather real data examples (Found in Section 3.1, 3.3).

- FNN_RecoverySimulation.R = This script contains the results for recovery beta(t) in Section 4.1.

- FNN_PredictionSimulation.R = This script contains the simulation predictions in Section 4.2.

- PackageVersions.lock = Contains the exact versions of all the packages loaded in FNN.R. This is for reproducibility purposes (see below) and this file opens in R.

- Data/bike.RData = Data for the bike example.

- Data/tecator.RDS = Data for the tecator example.

- Data/daily.RDS = Data for the weather example.

---------------------------------------------------------------

### Since we are using Keras/Tensorflow in the background, an installation of Python is required on the machine. Our results also require particular versions of these so, to make sure things work, we recommend the following steps (For Windows but added some instructions for MacOS):

1. Download Anaconda (https://www.anaconda.com/products/individual)

2. Open Anaconda Prompt (or, on Mac, open LaunchPad and click Terminal)

3. Create Python 3.7 environment

- conda create --name python37 python=3.7 activate

4. Install the Python version so Keras and Tensorflow 

- pip install tensorflow==1.14.0 

- pip install keras==2.2.4

5. Now, we need to open R and install the correct versions of Keras and Tensorflow for R, so in RStudio, run the following:

- require(devtools)

- install_version("tensorflow", version = "2.2.0", repos = "http://cran.us.r-project.org") 

- install_version("keras", version = "2.2.5.0", repos = "http://cran.us.r-project.org")

6. R wil likely default to the main version of Python you install when you installed Anaconda so you need to set it to the virtual environment we created earlier using Reticulate

- library(reticulate)

- use_condaenv(condaenv = 'PFDA', conda = "C:/~path_to_version/anaconda3/envs/Python37/python.exe")

- use_python("C:/~path_to_version/anaconda3/envs/Python37/python.exe")

- The above code has been commented out in the FNN.R file so you can uncomment and set the paths!

- As a side point, remember to delete .Rdata in your working directory and restart the R session before setting the above paths (otherwise R will default to that information)

7. At this point, everything should be working and you can just confirm you have the right versions of all the R packages by looking at the PackageVersions.Lock file!

- We recommend going into the FNN.R file and installing all the packages (preferrably the versions written out in the PackageVersions.Lock file)

- Remember, you need to change the "~path_to_version" to whatever the path is on your computer

- On a Mac, you'll have to set it to the appropriate version of Python which will not be an executable (.exe file)

- We recommend running FNN_Figure2.R script first as it runs quick relative to the other scripts. The simulation scripts take a very long time to run. All the scripts reproduce results found in the paper or the supplementary document!

- As far as the versions for R are concerned, we tested on R Version: 4.0.5 and R Studio Version: 1.4.1106

If there are any issues, feel free to contact us (you may also need to install the r-versions [not to be confused with R] of Keras and Tensorflow in Anaconda as well which has the versions r-keras 2.2.4.1 & r-tensorflow 1.13.1, respectively.) and we will attempt to respond to you promptly. We apologize for any inconvenience but this is a neccesary evil for anyone running these infrastructures in an R environment! Thank you.

