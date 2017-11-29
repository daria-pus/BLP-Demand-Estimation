# BLP-Demand-Estimation
Method of structural demand estimation using random-coefficients logit model of [Berry, Levinsohn and Pakes (1995)](http://www.jstor.org/stable/2171802?seq=1#page_scan_tab_contents). With an example that replicates the results from [Nevo (2000b)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.334.9199&rep=rep1&type=pdf). Data files and variable description were borrowed from [Bronwyn Hall](https://eml.berkeley.edu/~bhhall/e220c/readme.html). 

I would like to thank Prof. Nevo for making his [MATLAB code](http://faculty.wcas.northwestern.edu/~ane686/supplements/rc_dc_code.htm) available, which this program is based on.

The program consists of the following files:
- **BLP_notes.pdf** - file that explains motivation to use BLP, necessary data, model primitives, and estimation steps.
- **BLP_demand.py** - main file that defines BLP class, and using Data class runs the model and outputs coefficient estimates, standard errors and value of objective function. 
- **data_Nevo.py** - file that defines Data class for Nevo (2000b) replication.
- **iv.mat**, **ps2.mat** -  data files for Nevo (2000b) replication.

To run the program:
1. Run **data_Nevo.py**
2. Run **BLP_demand.py**

Results:

	               Mean        SD      Income   Income^2       Age      Child
    Constant  -2.010197  0.558228    2.292376   0.000000  1.284481   0.000000
               0.327063  0.162589    1.209040   0.000000  0.631174   0.000000
       Price -62.748438  3.313584  588.675608 -30.210302  0.000000  11.053792
              14.810492  1.340932  270.580464  14.108625  0.000000   4.122544
       Sugar   0.116253 -0.005793   -0.385094   0.000000  0.052249   0.000000
               0.016040  0.013507    0.121525   0.000000  0.025993   0.000000
       Mushy   0.499535  0.093504    0.747860   0.000000 -1.353348   0.000000
               0.198620  0.185438    0.802441   0.000000  0.667079   0.000000
	GMM objective: 4.561516417264547
	Min-Dist R-squared: 0.4591044701135003
	Min-Dist weighted R-squared: 0.10118275148991385
    
Note: Results differ from those in Nevo (2000b). This is because this code uses a tight tolerance for the contraction mapping. It minimizes the GMM objective function to 4.56, while Nevo's matlab code minimizes to 14.9.

By modifying Data class in **data_Nevo.py**, one can estimate random-coefficients logit model for any other data set.
