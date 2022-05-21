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
   :members:

.. autoclass:: composer.Callback
   :members:

.. autoclass:: composer.DataSpec
   :members:

.. autoclass:: composer.Engine
   :members:

.. autoclass:: composer.Evaluator
   :members:

.. autoclass:: composer.Event
   :members:

.. autoclass:: composer.State
   :members:

.. autoclass:: composer.Time
   :members:

.. autoclass:: composer.TimeUnit
   :members:

.. autoclass:: composer.Timestamp
   :members:
