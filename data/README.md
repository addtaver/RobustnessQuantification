# Data

`UCI/` contains the datasets used in the experiments. They are read and preprocessed (discretisation of numerical features, integer coding, binarisation of targets) by `robustness_quantification.data.loading.load_dataset`, which accepts these names:

| name | file(s) in `UCI/` |
|---|---|
| `adult` | `adult.data`, `adult.test` |
| `Australian` | `australian.dat` |
| `bank` | `bank.csv` |
| `BCW` | `breast-cancer-wisconsin.data` |
| `german` | `german.data` |
| `Healthy_aging` | `NPHA-doctor-visits.csv` |
| `heart_disease` | `processed.cleveland.data` |
| `lymphography` | `lymphography.data` |
| `mushroom` | `agaricus-lepiota.data` |
| `nursery` | `nursery.data` |
| `primary_tumor` | `primary-tumor.data` |
| `solar_small`, `solar_big` | `flare.data1`, `flare.data2` |
| `soybean` | `backup-large.data`, `backup-large.test` |
| `spect_heart` | `SPECT.train`, `SPECT.test` |
| `student_math_bin` | `student-mat.csv` |
| `student_port_bin` | `student-por.csv` |
| `zoo` | `zoo.data` |

`extra/` (not tracked by git) holds additional datasets that are not used by any experiment in this repository.
