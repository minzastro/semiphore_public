# Calibration data

There are several calibration data files that are needed in order
for semiphore to work:

* General catalog properties, specified in *utils/params.py*:
  * Names of columns with photometric magnitudes (*BANDS* dictionary)
  * Magnitude limits (*LIMITS* dictionary)
  * Extinction coefficiends (*EXTINCTIONS* dictionary)
* SED mixture data. This is a joblib file, that hosts a dict of the
following structure:
  * z: redshift grid of size N
  * names: survey names, those are used to select parameters also from *utils.params* dictionaries
  * weights: SED weights $w_i$, an array of size NxS, where S is the number
  of SEDs.
  * sed: SED values $\mu_b$, an array of size NxSxB, where B is the number
  of bands fitted.
  * err: SED width values $\sigma_b$, an array of size NxSxB, where B is the number
  of bands fitted.
  * l_values: Likelihood values for each redshift bin, for debugging purposes only
  * iterations: number of fit iterations used for each redshift bin, for debugging purposes only
  * sizes: number of objects used per redshift bin, for debugging purposes only
  * input_file: input data file, for debugging purposes only
  * items for intermediate values, for debugging purposes only. The content of those items is explained elsewhere.
* $m_{star}$ fit parameters. This is a joblib file, that hosts a dict of the
following structure: keys are band names (built from the lowercase catalogue name
and magnitude column name from *utils.params.BANDS* dictionary), and values
are tuples of:
  * Magnitude bins
  * Fitted parameters of the equation (7), [a, b, n, h, s]
  * Redshift grids for every magnitude bin
  * Histogram of object redshift distribution used to derive parameters
