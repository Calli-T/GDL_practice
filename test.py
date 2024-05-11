import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_rocm())
print(tf.test.is_built_with_gpu_support())
print(tf.sysconfig.get_build_info())

'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''


class MyTest(tf.test.TestCase):

    def test_add_on_gpu(self):
        if not tf.test.is_built_with_rocm():
            self.skipTest("test is only applicable on GPU")

        with tf.device("GPU:0"):
            self.assertEqual(tf.math.add(1.0, 2.0), 3.0)


t = MyTest()
t.test_add_on_gpu()

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

if not len(tf.config.list_physical_devices('GPU')) > 0:
    raise Exception('tf.config.list_physical_devices(\'GPU\') returns empty list; no GPUs found')
print(tf.config.list_physical_devices())