composer.core
=============

.. This module is more manual because we reimport a lot of its members
   in top-level composer. Note that the automodule composer.core
   on its own won't actually generate any docs because all of its members are
   imported. Manually specifying the members in the automodule didn't
   seem to work, so we just explicitly list them using auto{function,class}.

.. automodule:: composer.core

.. autofunction:: composer.core.ensure_data_spec
.. autofunction:: composer.core.ensure_time
.. autofunction:: composer.core.ensure_evaluator
.. autoclass:: composer.core.Trace
.. autoclass:: composer.core.Precision

.. autoclass:: composer.Algorithm
.. autoclass:: composer.Callback
.. autoclass:: composer.DataSpec
.. autoclass:: composer.Engine
.. autoclass:: composer.Evaluator
.. autoclass:: composer.Event
.. autoclass:: composer.State
.. autoclass:: composer.Time
.. autoclass:: composer.TimeUnit
.. autoclass:: composer.Timestamp


