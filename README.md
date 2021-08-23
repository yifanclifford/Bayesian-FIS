# B-FIS
## Requirements

Python 3.x and the required packages:
- argparse
- pyyaml
- pytorch >= 1.7.0
- pytrec_eval

### Install pytrec_eval

We made few changes in pytrec_eval to support mean reciprocal rank cut. Please install pytrec_eval following the instructions below:

- download pytrec_eval from [repo](https://github.com/cvangysel/pytrec_eval/archive/refs/heads/master.zip)
- modify setup.py in pytrec_eval:
```python
REMOTE_TREC_EVAL_URI = 'https://github.com/yifanclifford/trec_eval/archive/refs/heads/mrr_cut.zip'
REMOTE_TREC_EVAL_TLD_NAME = 'trec_eval-mrr_cut'
```
- install pytrec_eval
```
python setup.py install
```


Bayesian Feature Interaction Selection