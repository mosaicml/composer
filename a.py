import logging

try:
    # Operation that might cause an exception
    assert 1 == 2
except ZeroDivisionError as e:
    logging.error('Zero Division Error: %s', e)
except Exception as e:
    logging.exception('OKay')
    logging.exception('General Exception: %s', e)
