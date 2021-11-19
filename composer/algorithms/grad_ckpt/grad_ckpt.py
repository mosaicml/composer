import torch
from composer.algorithms.grad_ckpt.solver import ArbitrarySolver
from composer.algorithms.grad_ckpt.graph import Segment

from composer.algorithms.grad_ckpt.utils import disable_dropout
from composer.algorithms.grad_ckpt.parser import parse_graph

import argparse
from composer.core.types import Algorithm, Event, Logger, State

torch.backends.cudnn.enabled = True

class Grad_ckpt(Algorithm):

    def __init__(self, ):
        self.hparams = GradCkptHparams()


    def match(self, event: Event, state: State) -> bool:

        return event == Event.TRAINING_START

    def apply(self, event: Event, state: State, logger: Logger) -> None:

        assert state.model is not None, 'We cannot apply GC to None'

        if event == Event.TRAINING_START:
            #disable_dropout(arch, state.model)
            with torch.no_grad():
                x = torch.rand([1,3,32,32])
                graph, source, target = parse_graph(state.model.module, x)

            solver = ArbitrarySolver()

            run_graph, best_cost = solver.solve(graph, source, target)
            run_segment = Segment(run_graph, source, target, do_checkpoint=True)
            parsed_segment = Segmnet(graph, source, target, do_checkpoint=False)
            
            state.model = run_segment #GC(state.model)

            
#from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.algorithms.algorithm_hparams import AlgorithmHparams

log = logging.getLogger(__name__)


@dataclass
class GradCkptHparams(AlgorithmHparams):

    def initialize_object(self):
        return Grad_ckpt(**asdict(self))
