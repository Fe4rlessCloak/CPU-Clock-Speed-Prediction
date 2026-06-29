# CPU Clock Speed Prediction

Predicting Intel CPU clock speeds (base, turbo, and boost frequencies) using linear regression.

## Data

The dataset is sourced from [intel-processors](https://github.com/toUpperCase78/intel-processors) by toUpperCase78. It includes specifications across multiple Intel processor families (Atom, Celeron, Core, Core Ultra, Pentium, Xeon, and others).

## Approach

Linear regression is used to model the relationship between processor features (cores, threads, lithography, TDP, cache, memory specs) and clock speed targets.

## Structure

```
.
├── data/           # CSV datasets by processor family
├── model/          # Model training and evaluation
└── README.md
```

## Setup

```bash
conda create -n data python=3.12 pandas scikit-learn
conda activate data
```

## Usage

```bash
python model/main.py
```

## License

Data attributed to [toUpperCase78/intel-processors](https://github.com/toUpperCase78/intel-processors). Code under MIT.
