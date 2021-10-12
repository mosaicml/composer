# Copyright 2021 MosaicML. All Rights Reserved.

from composer.algorithms.curriculum_learning.curriculum_learning import CurriculumLearning as CurriculumLearning
from composer.algorithms.curriculum_learning.curriculum_learning import \
    CurriculumLearningHparams as CurriculumLearningHparams

_name = 'Curriculum Learning'
_class_name = 'CurriculumLearning'
_functional = 'apply_curriculum'
_tldr = 'Using sequence length as a proxy for example difficulty, it warms up the sequence length for a specified duration of training.'
_attribution = '(Li et al, 2021)'
_link = 'https://arxiv.org/abs/2108.06084'
_method_card = ''
