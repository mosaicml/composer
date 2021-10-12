from __future__ import annotations

import abc
from dataclasses import dataclass

import yahp as hp

from composer.core.callback import Callback


@dataclass
class CallbackHparams(hp.Hparams, abc.ABC):

    @abc.abstractmethod
    def initialize_object(self) -> Callback:
        pass
