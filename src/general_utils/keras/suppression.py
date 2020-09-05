def suppress_tf_deprecation_messages():
    # suppress deprecation warnings in current thread
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False


def suppress_tf_warnings_and_info():
    import os
    # suppress Tensorflow warnings and info in current thread
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
