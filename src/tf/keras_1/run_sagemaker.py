import sagemaker
import os
from sagemaker.utils import sagemaker_timestamp
from sagemaker.tensorflow import TensorFlow
from sagemaker import get_execution_role


sagemaker_session = sagemaker.Session()

default_s3_bucket = sagemaker_session.default_bucket()
sagemaker_iam_role = get_execution_role()

train_script = "mnist_hvd.py"
instance_count = 2