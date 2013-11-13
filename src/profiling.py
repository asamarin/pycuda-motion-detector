#!/opt/python2.7/bin/python
import cProfile

class Profile:

    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        cProfile.runctx('self.f(self.f.__class__, *args)', globals(), locals())
