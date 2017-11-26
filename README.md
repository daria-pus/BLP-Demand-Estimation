# BLP-Demand-Estimation
Method of structural demand estimation using random-coefficients logit model of [Berry, Levinsohn and Pakes (1995)](http://www.jstor.org/stable/2171802?seq=1#page_scan_tab_contents). With an example that replicates the results from [Nevo (2000b)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.334.9199&rep=rep1&type=pdf). Data files and variable description were borrowed from [Bronwyn Hall](https://eml.berkeley.edu/~bhhall/e220c/readme.html). 

I would like to thank Prof. Nevo for making his [MATLAB code](http://faculty.wcas.northwestern.edu/~ane686/supplements/rc_dc_code.htm) available, which this program is based on.

The program consists of the following files:
- **BLP_demand.py** - main file that defines BLP class, and using Data class runs the model and outputs coefficient estimates, standard errors and value of objective function. 
- **data_Nevo.py** - file that defines Data class for Nevo (2000b) replication.
- **iv.mat**, **ps2.mat** -  data files for Nevo (2000b) replication.

To run the program:
1. Run **data_Nevo.py**
2. Run **BLP_demand.py**

By modifying Data class in **data_Nevo.py**, one can estimate random-coefficients logit model for any other data set.