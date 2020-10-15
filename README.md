# Covid-19 Adverse Outcome Risk Predection Models

This Python implementation allows for testing the models from the paper:

  [__DEVELOPING AND VALIDATING COVID-19 ADVERSE OUTCOME RISK PREDICTION MODELS FROM A BI-NATIONAL EUROPEAN COHORT OF 5594 PATIENTS__](https://www.medrxiv.org/content/10.1101/2020.10.06.20207209v1)<br />
  <sub>Espen Jimenez Solem, Tonny Studsgaard Petersen, Christina Lioma, Christian Igel, Wouter Boomsma, Oswin Krause, Casper Hansen, Christian Hansen, Stephan Lorentzen, Raghavendra Selvan, Janne Petersen, Martin Erik Nyeland, Mikkel Zoellner Ankarfeldt, Gert Mehl Virenfeldt, Mathilde Winther-Jensen, Allan Linneberg, Mostafa Mediphour Ghazi, Nicki Detlefsen, Andreas Lauritzen, Abraham George Smith, Marleen de Bruijne, Bulat Ibragimov, Jens Petersen, Martin Lillholm, Marie Helleberg, Benjamin Skov Kaas-Hansen, Jon Middleton, Stine Hasling Mogensen, Hans Christian Thorsen-Meyer, Anders Perner, Mikkel Bonde, Alexander Bonde, Akshay Pai, Mads Nielsen, Martin Sillesen</sub>

and generating syntetic data for tests. The models trained on the real data is included with the implementation.


## Requirements

The implemenation is tested using Python 3.8.5, and requires the following
libraries:
* scikit-learn
* numpy
* pandas
* pickle
* joblib



## Generating data

To generate a data set, run:

```
python generator.py <N>
```

where N is the number of patients. This generates `data/patients.csv` according to the distribution in `data/demographics.csv` (which can be edited).



## Testing model

To test a model, run:

```
python cope.py [-h] [-t T] [-p P] [-f FS] data
```

T is the time of prediction:
* `test`:     at time of Covid19 test,
* `hospital`: 12h after hospitalization,
* `pre-icu`:  12h before icu admission,
* `post-icu`: 12h after icu admission.

P is the prediction target: `hospital`, `icu`, `ventilator` or `death`.

FS is the feature set (see section below for more details):
* `basic`:         BMI+age+sex,
* `comorbidities`: adds comorbidities,
* `temporal`:      adds temporal features,
* `tests`:         adds in-hospital tests.

`data` is the path to the input data file.

E.g. to run the model for predicting death among hospitalized patients using all available features (and using the data generated by `generator.py`):

```
python cope -t hospital -p death -f tests data/patients.csv
```


## Model Overview

The models are located in the `models` directory, and named using the following scheme:

```
models/<data-set>_rf_<feature-list>_<target>.model
```

where `<data-set>` is one of `covid19_infected_0`,`hospital_admission_12`,`icu_admission_-12`,`icu_admission_12`, `<feature-list>` is a prefix of `basicinfo_disease_temporal_features_tests`,
and `<target>` is one of `death`,`used_ventilator`,`ICU_admitted`,`hospital_admitted`.

The models can be loaded using joblib:
```
joblib.load("models/<data-set>_rf_<feature-list>_<target>.model")
```
The files:
```
models/<data-set>_rf_<feature-list>_<target>_dtypes_colnames.pkl
```
list the input features in the correct order for the corresponding model, and can be loaded using pickle.



## Feature overview
Below follows overview of the four different feature sets. Each set consists of
all listed features in addition to the features of any of the sets listed above:

### basic
```
BMI
is_male
age
```

### comorbidities
```
diabetes
ischemic_heart_disease
heart_failure
arrhythmia
stroke
COPD_asthma
arthritis
osteoporosis
dementia
severe_mental_disorder
immunodeficiencies
neurological_manifestations
cancer
chronic_kidney_failure
dialysis
hypertension
```

### temporal
```
hours_diagnosis_to_hospitalization
hours_diagnosis_to_ICU
hours_hospitalization_to_ICU
```

### tests
```
crp_mean
crp_count
crp_slope
lymphopaenia_mean
lymphopaenia_count
lymphopaenia_slope
lactic_dehydrogenase_mean
lactic_dehydrogenase_count
lactic_dehydrogenase_slope
alanine_aminotransferase_mean
alanine_aminotransferase_count
alanine_aminotransferase_slope
red_blood_cells_mean
red_blood_cells_count
red_blood_cells_slope
white_blood_cells_mean
white_blood_cells_count
white_blood_cells_slope
neutrophil_count_mean
neutrophil_count_count
neutrophil_count_slope
d_dimer_mean
d_dimer_count
d_dimer_slope
blood_urea_nitrogen_mean
blood_urea_nitrogen_count
blood_urea_nitrogen_slope
creatinine_mean
creatinine_count
creatinine_slope
ferritin_mean
ferritin_count
ferritin_slope
base_excess_mean
base_excess_count
base_excess_slope
hydrogencarbonat_mean
hydrogencarbonat_count
hydrogencarbonat_slope
laktat_mean
laktat_count
laktat_slope
o2_mean
o2_count
o2_slope
pco2_mean
pco2_count
pco2_slope
ph_mean
ph_count
ph_slope
po2_mean
po2_count
po2_slope
crp_most_recent
lymphopaenia_most_recent
lactic_dehydrogenase_most_recent
alanine_aminotransferase_most_recent
red_blood_cells_most_recent
white_blood_cells_most_recent
neutrophil_count_most_recent
d_dimer_most_recent
blood_urea_nitrogen_most_recent
creatinine_most_recent
ferritin_most_recent
base_excess_most_recent
hydrogencarbonat_most_recent
laktat_most_recent
o2_most_recent
pco2_most_recent
ph_most_recent
po2_most_recent
pulse_mean
pulse_count
pulse_slope
temp_mean
temp_count
temp_slope
EWS_mean
EWS_count
EWS_slope
respiratory_rate_mean
respiratory_rate_count
respiratory_rate_slope
saturation_mean
saturation_count
saturation_slope
pulse_most_recent
temp_most_recent
EWS_most_recent
respiratory_rate_most_recent
saturation_most_recent
is_smoking
```
